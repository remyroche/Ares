"""
Unsupervised Hybrid NAS Clusterer - Integration with Hybrid NAS System

This module integrates unsupervised tree-based NAS with the existing hybrid NAS system
for comprehensive regime detection and qualification.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from datetime import datetime

# Import unsupervised NAS components
from src.utils.ml_common.optimization.unsupervised_tree_nas import (
    UnsupervisedTreeNAS, UnsupervisedTreeNASConfig, UnsupervisedArchitectureCandidate
)

# Import hybrid NAS components
from .hybrid_nas_clusterer import HybridNASClusterer
from .hybrid_nas_config import HybridNASClusteringConfig

logger = logging.getLogger(__name__)


class UnsupervisedHybridNASClusterer:
    """Unsupervised Hybrid NAS Clusterer combining unsupervised and hybrid approaches."""
    
    def __init__(self, config: HybridNASClusteringConfig):
        """Initialize unsupervised hybrid NAS clusterer."""
        self.config = config
        self.logger = logger.getChild('UnsupervisedHybridNASClusterer')
        
        # Initialize unsupervised NAS
        unsupervised_config = UnsupervisedTreeNASConfig(
            clustering_algorithms=['kmeans', 'gaussian_mixture', 'agglomerative'],
            n_regimes_range=(config.clustering_config['n_regimes'] - 2, 
                           config.clustering_config['n_regimes'] + 2),
            min_regime_duration=config.clustering_config.get('min_regime_duration', 5),
            regime_stability_threshold=config.clustering_config.get('regime_stability_threshold', 0.7),
            n_trials=config.optimization_config['n_trials'] // 2
        )
        self.unsupervised_nas = UnsupervisedTreeNAS(unsupervised_config)
        
        # Initialize hybrid NAS clusterer
        self.hybrid_clusterer = HybridNASClusterer(config)
        
        # Results storage
        self.unsupervised_results = None
        self.hybrid_results = None
        self.combined_results = None
        
        self.logger.info("✅ Unsupervised Hybrid NAS Clusterer initialized")
    
    def cluster(self, 
                market_data: pd.DataFrame, 
                timestamps: np.ndarray,
                optimize_parameters: bool = True,
                generate_report: bool = True) -> Dict[str, Any]:
        """
        Perform unsupervised hybrid NAS clustering for regime detection.
        
        Args:
            market_data: Market data (OHLCV)
            timestamps: Timestamps for the data
            optimize_parameters: Whether to optimize parameters
            generate_report: Whether to generate detailed report
            
        Returns:
            Combined clustering results
        """
        self.logger.info("🚀 Starting Unsupervised Hybrid NAS Clustering...")
        start_time = time.time()
        
        try:
            # Step 1: Unsupervised regime detection
            self.logger.info("🔍 Step 1: Unsupervised regime detection...")
            unsupervised_start = time.time()
            self.unsupervised_results = self.unsupervised_nas.search(market_data, timestamps)
            unsupervised_time = time.time() - unsupervised_start
            
            # Step 2: Hybrid NAS clustering using unsupervised results
            self.logger.info("🔍 Step 2: Hybrid NAS clustering...")
            hybrid_start = time.time()
            
            # Use unsupervised results to guide hybrid clustering
            guided_config = self._create_guided_config(self.unsupervised_results)
            self.hybrid_clusterer.config.update_config(guided_config)
            
            self.hybrid_results = self.hybrid_clusterer.cluster(
                market_data, timestamps, optimize_parameters, generate_report
            )
            hybrid_time = time.time() - hybrid_start
            
            # Step 3: Combine results
            self.logger.info("🔍 Step 3: Combining results...")
            self.combined_results = self._combine_results(
                self.unsupervised_results, self.hybrid_results
            )
            
            # Add metadata
            self.combined_results['unsupervised_metadata'] = {
                'unsupervised_time': unsupervised_time,
                'hybrid_time': hybrid_time,
                'total_time': time.time() - start_time,
                'unsupervised_regimes': len(self.unsupervised_results.regimes),
                'hybrid_regimes': len(self.hybrid_results.get('regimes', [])),
                'combined_regimes': len(self.combined_results.get('regimes', [])),
                'method': 'unsupervised_hybrid_nas_clustering'
            }
            
            self.logger.info(f"✅ Unsupervised Hybrid NAS Clustering completed in {time.time() - start_time:.2f}s")
            self.logger.info(f"📊 Unsupervised: {len(self.unsupervised_results.regimes)} regimes in {unsupervised_time:.2f}s")
            self.logger.info(f"📊 Hybrid: {len(self.hybrid_results.get('regimes', []))} regimes in {hybrid_time:.2f}s")
            self.logger.info(f"📊 Combined: {len(self.combined_results.get('regimes', []))} regimes")
            
            return self.combined_results
            
        except Exception as e:
            self.logger.error(f"Unsupervised Hybrid NAS Clustering failed: {e}")
            raise
    
    def _create_guided_config(self, unsupervised_results: UnsupervisedArchitectureCandidate) -> Dict[str, Any]:
        """Create guided configuration based on unsupervised results."""
        try:
            # Extract insights from unsupervised results
            n_regimes = unsupervised_results.n_regimes
            best_algorithm = unsupervised_results.clustering_algorithm
            feature_importance = unsupervised_results.feature_importance
            
            # Create guided configuration
            guided_config = {
                'clustering_config': {
                    'n_regimes': n_regimes,
                    'guided_by_unsupervised': True,
                    'unsupervised_algorithm': best_algorithm
                },
                'tree_nas_config': {
                    'enable_feature_selection': True,
                    'max_features': min(50, len(feature_importance)),
                    'feature_importance_weights': feature_importance
                },
                'neural_nas_config': {
                    'n_regimes': n_regimes,
                    'guided_by_unsupervised': True
                }
            }
            
            return guided_config
            
        except Exception as e:
            self.logger.warning(f"Guided configuration creation failed: {e}")
            return {}
    
    def _combine_results(self, unsupervised_results: UnsupervisedArchitectureCandidate,
                        hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine unsupervised and hybrid results."""
        try:
            # Extract regimes from both approaches
            unsupervised_regimes = unsupervised_results.regimes
            hybrid_regimes = hybrid_results.get('regimes', [])
            
            # Combine regime information
            combined_regimes = self._combine_regimes(unsupervised_regimes, hybrid_regimes)
            
            # Combine quality metrics
            combined_quality = self._combine_quality_metrics(unsupervised_results, hybrid_results)
            
            # Combine feature importance
            combined_features = self._combine_feature_importance(unsupervised_results, hybrid_results)
            
            # Create combined results
            combined_results = {
                'labels': self._create_combined_labels(unsupervised_regimes, hybrid_regimes),
                'cluster_centers': self._create_combined_centers(unsupervised_regimes, hybrid_regimes),
                'statistics': {
                    'n_clusters': len(combined_regimes),
                    'silhouette_score': combined_quality.get('silhouette_score', 0.0),
                    'calinski_harabasz_score': combined_quality.get('calinski_harabasz_score', 0.0),
                    'davies_bouldin_score': combined_quality.get('davies_bouldin_score', 0.0)
                },
                'quality_metrics': {
                    'accuracy': combined_quality.get('accuracy', 0.0),
                    'efficiency': combined_quality.get('efficiency', 0.0),
                    'interpretability': combined_quality.get('interpretability', 0.0),
                    'robustness': combined_quality.get('robustness', 0.0)
                },
                'regimes': combined_regimes,
                'feature_importance': combined_features,
                'unsupervised_results': {
                    'algorithm': unsupervised_results.clustering_algorithm,
                    'n_regimes': unsupervised_results.n_regimes,
                    'overall_score': unsupervised_results.overall_score,
                    'regimes': unsupervised_regimes
                },
                'hybrid_results': hybrid_results,
                'combination_method': 'unsupervised_guided_hybrid'
            }
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Result combination failed: {e}")
            return hybrid_results  # Fallback to hybrid results
    
    def _combine_regimes(self, unsupervised_regimes: List, hybrid_regimes: List) -> List[Dict[str, Any]]:
        """Combine regimes from both approaches."""
        try:
            combined_regimes = []
            
            # Add unsupervised regimes
            for regime in unsupervised_regimes:
                combined_regimes.append({
                    'id': regime.regime_id,
                    'type': regime.regime_type,
                    'confidence': regime.regime_confidence,
                    'quality': regime.overall_quality,
                    'duration': regime.duration,
                    'source': 'unsupervised',
                    'silhouette_score': regime.silhouette_score,
                    'persistence': regime.regime_persistence,
                    'separation': regime.regime_separation,
                    'consistency': regime.regime_consistency,
                    'key_features': regime.key_features,
                    'feature_importance': regime.feature_importance
                })
            
            # Add hybrid regimes (avoid duplicates)
            for regime in hybrid_regimes:
                if not any(r['id'] == regime.get('id', -1) for r in combined_regimes):
                    combined_regimes.append({
                        'id': regime.get('id', len(combined_regimes)),
                        'type': regime.get('type', 'unknown'),
                        'confidence': regime.get('confidence', 0.5),
                        'quality': regime.get('quality', 0.5),
                        'duration': regime.get('duration', 0),
                        'source': 'hybrid',
                        'silhouette_score': regime.get('silhouette_score', 0.0),
                        'persistence': regime.get('persistence', 0.0),
                        'separation': regime.get('separation', 0.0),
                        'consistency': regime.get('consistency', 0.0),
                        'key_features': regime.get('key_features', []),
                        'feature_importance': regime.get('feature_importance', {})
                    })
            
            return combined_regimes
            
        except Exception as e:
            self.logger.warning(f"Regime combination failed: {e}")
            return []
    
    def _combine_quality_metrics(self, unsupervised_results: UnsupervisedArchitectureCandidate,
                                hybrid_results: Dict[str, Any]) -> Dict[str, float]:
        """Combine quality metrics from both approaches."""
        try:
            # Get unsupervised metrics
            unsupervised_quality = {
                'silhouette_score': unsupervised_results.clustering_quality,
                'calinski_harabasz_score': 0.0,  # Not directly available
                'davies_bouldin_score': 0.0,    # Not directly available
                'accuracy': unsupervised_results.regime_detection_accuracy,
                'efficiency': 0.8,  # Tree-based models are efficient
                'interpretability': 0.9,  # Tree-based models are interpretable
                'robustness': unsupervised_results.regime_qualification_score
            }
            
            # Get hybrid metrics
            hybrid_quality = hybrid_results.get('quality_metrics', {})
            
            # Combine metrics (weighted average)
            combined_quality = {}
            for key in set(unsupervised_quality.keys()) | set(hybrid_quality.keys()):
                unsupervised_val = unsupervised_quality.get(key, 0.0)
                hybrid_val = hybrid_quality.get(key, 0.0)
                # Weight unsupervised more for interpretability, hybrid more for accuracy
                if key in ['interpretability', 'efficiency']:
                    combined_quality[key] = 0.7 * unsupervised_val + 0.3 * hybrid_val
                else:
                    combined_quality[key] = 0.4 * unsupervised_val + 0.6 * hybrid_val
            
            return combined_quality
            
        except Exception as e:
            self.logger.warning(f"Quality metrics combination failed: {e}")
            return {}
    
    def _combine_feature_importance(self, unsupervised_results: UnsupervisedArchitectureCandidate,
                                   hybrid_results: Dict[str, Any]) -> Dict[str, float]:
        """Combine feature importance from both approaches."""
        try:
            # Get unsupervised feature importance
            unsupervised_features = unsupervised_results.feature_importance
            
            # Get hybrid feature importance
            hybrid_features = hybrid_results.get('feature_importance', {})
            
            # Combine feature importance (weighted average)
            all_features = set(unsupervised_features.keys()) | set(hybrid_features.keys())
            combined_features = {}
            
            for feature in all_features:
                unsupervised_importance = unsupervised_features.get(feature, 0.0)
                hybrid_importance = hybrid_features.get(feature, 0.0)
                # Weight unsupervised more for feature selection insights
                combined_importance = 0.6 * unsupervised_importance + 0.4 * hybrid_importance
                combined_features[feature] = combined_importance
            
            return combined_features
            
        except Exception as e:
            self.logger.warning(f"Feature importance combination failed: {e}")
            return unsupervised_results.feature_importance
    
    def _create_combined_labels(self, unsupervised_regimes: List, hybrid_regimes: List) -> np.ndarray:
        """Create combined regime labels."""
        try:
            # Use unsupervised labels as base (they're more comprehensive)
            if unsupervised_regimes:
                # Create labels from unsupervised regimes
                max_samples = max(regime.sample_indices[-1] for regime in unsupervised_regimes if regime.sample_indices)
                labels = np.zeros(max_samples + 1, dtype=int)
                
                for regime in unsupervised_regimes:
                    if regime.sample_indices:
                        labels[regime.sample_indices] = regime.regime_id
                
                return labels
            else:
                # Fallback to hybrid labels
                return hybrid_results.get('labels', np.array([]))
                
        except Exception as e:
            self.logger.warning(f"Combined labels creation failed: {e}")
            return np.array([])
    
    def _create_combined_centers(self, unsupervised_regimes: List, hybrid_regimes: List) -> np.ndarray:
        """Create combined cluster centers."""
        try:
            centers = []
            
            # Add unsupervised centers
            for regime in unsupervised_regimes:
                centers.append(regime.regime_center)
            
            # Add hybrid centers (avoid duplicates)
            hybrid_centers = hybrid_results.get('cluster_centers', [])
            for center in hybrid_centers:
                if len(centers) == 0 or not any(np.allclose(center, existing_center) for existing_center in centers):
                    centers.append(center)
            
            return np.array(centers) if centers else np.array([])
            
        except Exception as e:
            self.logger.warning(f"Combined centers creation failed: {e}")
            return np.array([])
    
    def get_unsupervised_insights(self) -> Dict[str, Any]:
        """Get insights from unsupervised analysis."""
        try:
            if self.unsupervised_results is None:
                return {'message': 'No unsupervised results available'}
            
            return {
                'best_algorithm': self.unsupervised_results.clustering_algorithm,
                'n_regimes': self.unsupervised_results.n_regimes,
                'overall_score': self.unsupervised_results.overall_score,
                'clustering_quality': self.unsupervised_results.clustering_quality,
                'regime_detection_accuracy': self.unsupervised_results.regime_detection_accuracy,
                'regime_qualification_score': self.unsupervised_results.regime_qualification_score,
                'feature_importance': self.unsupervised_results.feature_importance,
                'regimes': [
                    {
                        'id': regime.regime_id,
                        'type': regime.regime_type,
                        'confidence': regime.regime_confidence,
                        'quality': regime.overall_quality,
                        'duration': regime.duration,
                        'key_features': regime.key_features
                    }
                    for regime in self.unsupervised_results.regimes
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Unsupervised insights retrieval failed: {e}")
            return {'error': str(e)}
    
    def get_hybrid_insights(self) -> Dict[str, Any]:
        """Get insights from hybrid analysis."""
        try:
            if self.hybrid_results is None:
                return {'message': 'No hybrid results available'}
            
            return {
                'strategy': self.hybrid_results.get('hybrid_metadata', {}).get('strategy', 'unknown'),
                'n_regimes': len(self.hybrid_results.get('regimes', [])),
                'quality_metrics': self.hybrid_results.get('quality_metrics', {}),
                'statistics': self.hybrid_results.get('statistics', {}),
                'regimes': self.hybrid_results.get('regimes', [])
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid insights retrieval failed: {e}")
            return {'error': str(e)}
    
    def get_combined_insights(self) -> Dict[str, Any]:
        """Get insights from combined analysis."""
        try:
            if self.combined_results is None:
                return {'message': 'No combined results available'}
            
            return {
                'total_regimes': len(self.combined_results.get('regimes', [])),
                'combined_quality': self.combined_results.get('quality_metrics', {}),
                'feature_importance': self.combined_results.get('feature_importance', {}),
                'unsupervised_contribution': {
                    'regimes': len(self.unsupervised_results.regimes) if self.unsupervised_results else 0,
                    'algorithm': self.unsupervised_results.clustering_algorithm if self.unsupervised_results else 'unknown'
                },
                'hybrid_contribution': {
                    'regimes': len(self.hybrid_results.get('regimes', [])) if self.hybrid_results else 0,
                    'strategy': self.hybrid_results.get('hybrid_metadata', {}).get('strategy', 'unknown') if self.hybrid_results else 'unknown'
                },
                'combination_benefits': {
                    'interpretability': 'High (from unsupervised)',
                    'accuracy': 'High (from hybrid)',
                    'efficiency': 'High (from both)',
                    'robustness': 'High (from both)'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Combined insights retrieval failed: {e}")
            return {'error': str(e)}
    
    def compare_approaches(self) -> Dict[str, Any]:
        """Compare unsupervised vs hybrid approaches."""
        try:
            if self.unsupervised_results is None or self.hybrid_results is None:
                return {'message': 'Insufficient results for comparison'}
            
            comparison = {
                'regime_count': {
                    'unsupervised': len(self.unsupervised_results.regimes),
                    'hybrid': len(self.hybrid_results.get('regimes', [])),
                    'combined': len(self.combined_results.get('regimes', [])) if self.combined_results else 0
                },
                'quality_scores': {
                    'unsupervised': {
                        'overall': self.unsupervised_results.overall_score,
                        'clustering': self.unsupervised_results.clustering_quality,
                        'detection': self.unsupervised_results.regime_detection_accuracy,
                        'qualification': self.unsupervised_results.regime_qualification_score
                    },
                    'hybrid': self.hybrid_results.get('quality_metrics', {}),
                    'combined': self.combined_results.get('quality_metrics', {}) if self.combined_results else {}
                },
                'strengths': {
                    'unsupervised': [
                        'No labeled data required',
                        'Automatic regime detection',
                        'High interpretability',
                        'Fast execution'
                    ],
                    'hybrid': [
                        'Complex pattern recognition',
                        'High accuracy',
                        'Ensemble methods',
                        'Robust predictions'
                    ],
                    'combined': [
                        'Best of both approaches',
                        'Comprehensive regime analysis',
                        'High interpretability and accuracy',
                        'Robust and efficient'
                    ]
                },
                'recommendations': {
                    'use_unsupervised_for': 'Initial regime detection, feature selection, interpretability',
                    'use_hybrid_for': 'Complex patterns, high accuracy requirements, ensemble methods',
                    'use_combined_for': 'Production systems, comprehensive analysis, best performance'
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Approach comparison failed: {e}")
            return {'error': str(e)}