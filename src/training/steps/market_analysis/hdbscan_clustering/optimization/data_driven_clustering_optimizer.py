"""
Data-Driven Clustering Parameter Optimizer

This module provides the main optimizer that integrates all data-driven parameter
optimization components: feature weights, merging thresholds, temporal windows,
and validation thresholds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
import time
import os
from pathlib import Path

# Import individual optimizers
from .data_driven_feature_weights import DataDrivenFeatureWeightOptimizer, FeatureGroupWeightResult
from .data_driven_merging_thresholds import DataDrivenMergingThresholdOptimizer, RegimeMergingThresholdResult
from .data_driven_temporal_windows import DataDrivenTemporalWindowOptimizer, TemporalWindowResult
from .data_driven_validation_thresholds import DataDrivenValidationThresholdOptimizer, ClusterValidationThresholdResult

# Import configuration
from ..config.data_driven_config import DataDrivenClusteringConfig

logger = logging.getLogger(__name__)


@dataclass
class DataDrivenClusteringResult:
    """Result of comprehensive data-driven clustering optimization."""
    feature_weights_result: Optional[FeatureGroupWeightResult]
    merging_thresholds_result: Optional[RegimeMergingThresholdResult]
    temporal_windows_result: Optional[TemporalWindowResult]
    validation_thresholds_result: Optional[ClusterValidationThresholdResult]
    
    # Combined results
    optimal_parameters: Dict[str, Any]
    overall_score: float
    optimization_summary: Dict[str, Any]
    
    # Metadata
    optimization_time: float
    convergence_info: Dict[str, Any]
    metadata: Dict[str, Any]


class DataDrivenClusteringOptimizer:
    """
    Main optimizer for data-driven clustering parameters.
    
    Integrates all individual optimizers to replace hardcoded parameters with
    data-driven alternatives based on clustering quality and economic metrics.
    """
    
    def __init__(self, config: DataDrivenClusteringConfig):
        """
        Initialize the data-driven clustering optimizer.
        
        Args:
            config: Configuration for data-driven optimization
        """
        self.config = config
        self.optimization_history = []
        self.start_time = None
        
        # Initialize individual optimizers
        self.feature_weight_optimizer = DataDrivenFeatureWeightOptimizer(config.feature_weights)
        self.merging_threshold_optimizer = DataDrivenMergingThresholdOptimizer(config.merging_thresholds)
        self.temporal_window_optimizer = DataDrivenTemporalWindowOptimizer(config.temporal_windows)
        self.validation_threshold_optimizer = DataDrivenValidationThresholdOptimizer(config.validation_thresholds)
        
    def optimize_all_parameters(self, 
                               market_data: pd.DataFrame,
                               features: np.ndarray,
                               feature_names: List[str],
                               clustering_func: Callable,
                               economic_validation_func: Optional[Callable] = None) -> DataDrivenClusteringResult:
        """
        Optimize all clustering parameters using data-driven methods.
        
        Args:
            market_data: Market data for analysis
            features: Feature matrix
            feature_names: List of feature names
            clustering_func: Function that performs clustering
            economic_validation_func: Optional function for economic validation
            
        Returns:
            DataDrivenClusteringResult with all optimized parameters
        """
        try:
            self.start_time = time.time()
            logger.info("🚀 Starting comprehensive data-driven clustering optimization...")
            
            # Validate configuration
            self.config.validate()
            
            # Initialize results
            results = {}
            optimal_parameters = {}
            
            # Run optimization in specified order
            for optimization_type in self.config.optimization_order:
                logger.info(f"🔄 Optimizing {optimization_type}...")
                
                if optimization_type == 'feature_weights' and self.config.feature_weights.enable_optimization:
                    result = self._optimize_feature_weights(
                        features, feature_names, market_data, clustering_func, economic_validation_func
                    )
                    results['feature_weights'] = result
                    optimal_parameters.update(result.optimal_weights)
                    
                elif optimization_type == 'temporal_windows' and self.config.temporal_windows.enable_optimization:
                    result = self._optimize_temporal_windows(
                        market_data, clustering_func, economic_validation_func
                    )
                    results['temporal_windows'] = result
                    optimal_parameters.update(result.optimal_windows)
                    
                elif optimization_type == 'merging_thresholds' and self.config.merging_thresholds.enable_optimization:
                    result = self._optimize_merging_thresholds(
                        features, clustering_func
                    )
                    results['merging_thresholds'] = result
                    optimal_parameters.update(result.optimal_thresholds)
                    
                elif optimization_type == 'validation_thresholds' and self.config.validation_thresholds.enable_optimization:
                    result = self._optimize_validation_thresholds(
                        features, clustering_func, economic_validation_func
                    )
                    results['validation_thresholds'] = result
                    optimal_parameters.update(result.optimal_thresholds)
                
                logger.info(f"✅ {optimization_type} optimization completed")
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(results)
            
            # Create optimization summary
            optimization_summary = self._create_optimization_summary(results, overall_score)
            
            # Calculate convergence info
            convergence_info = self._calculate_convergence_info(results)
            
            # Create final result
            final_result = DataDrivenClusteringResult(
                feature_weights_result=results.get('feature_weights'),
                merging_thresholds_result=results.get('merging_thresholds'),
                temporal_windows_result=results.get('temporal_windows'),
                validation_thresholds_result=results.get('validation_thresholds'),
                optimal_parameters=optimal_parameters,
                overall_score=overall_score,
                optimization_summary=optimization_summary,
                optimization_time=time.time() - self.start_time,
                convergence_info=convergence_info,
                metadata={
                    'config': self.config.to_dict(),
                    'n_samples': features.shape[0],
                    'n_features': features.shape[1],
                    'optimization_order': self.config.optimization_order
                }
            )
            
            # Save results if configured
            if self.config.save_optimization_results:
                self._save_optimization_results(final_result)
            
            logger.info(f"🎉 Data-driven clustering optimization completed in {final_result.optimization_time:.2f}s")
            logger.info(f"📊 Overall score: {overall_score:.4f}")
            logger.info(f"📈 Optimal parameters: {optimal_parameters}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Data-driven clustering optimization failed: {e}")
            raise
    
    def _optimize_feature_weights(self, 
                                 features: np.ndarray,
                                 feature_names: List[str],
                                 market_data: pd.DataFrame,
                                 clustering_func: Callable,
                                 economic_validation_func: Optional[Callable]) -> FeatureGroupWeightResult:
        """Optimize feature group weights."""
        try:
            # Create wrapper function that applies weights
            def weighted_clustering_func(weighted_features):
                return clustering_func(weighted_features)
            
            return self.feature_weight_optimizer.optimize_weights(
                features, feature_names, market_data, weighted_clustering_func, economic_validation_func
            )
        except Exception as e:
            logger.error(f"Feature weight optimization failed: {e}")
            raise
    
    def _optimize_temporal_windows(self, 
                                  market_data: pd.DataFrame,
                                  clustering_func: Callable,
                                  economic_validation_func: Optional[Callable]) -> TemporalWindowResult:
        """Optimize temporal window sizes."""
        try:
            # Create wrapper function that accepts window parameters
            def windowed_clustering_func(data, windows):
                return clustering_func(data, windows)
            
            return self.temporal_window_optimizer.optimize_windows(
                market_data, windowed_clustering_func, economic_validation_func
            )
        except Exception as e:
            logger.error(f"Temporal window optimization failed: {e}")
            raise
    
    def _optimize_merging_thresholds(self, 
                                   features: np.ndarray,
                                   clustering_func: Callable) -> RegimeMergingThresholdResult:
        """Optimize regime merging thresholds."""
        try:
            # Perform initial clustering
            initial_labels = clustering_func(features)
            
            # Create wrapper function for merging
            def merging_func(labels, features, thresholds):
                # This would typically call a merging function with the thresholds
                # For now, return the original labels
                return labels
            
            return self.merging_threshold_optimizer.optimize_thresholds(
                initial_labels, features, merging_func
            )
        except Exception as e:
            logger.error(f"Merging threshold optimization failed: {e}")
            raise
    
    def _optimize_validation_thresholds(self, 
                                      features: np.ndarray,
                                      clustering_func: Callable,
                                      economic_validation_func: Optional[Callable]) -> ClusterValidationThresholdResult:
        """Optimize cluster validation thresholds."""
        try:
            return self.validation_threshold_optimizer.optimize_thresholds(
                features, clustering_func, economic_validation_func
            )
        except Exception as e:
            logger.error(f"Validation threshold optimization failed: {e}")
            raise
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall optimization score."""
        try:
            scores = []
            weights = []
            
            # Feature weights score
            if 'feature_weights' in results and results['feature_weights']:
                scores.append(results['feature_weights'].optimization_score)
                weights.append(0.3)
            
            # Temporal windows score
            if 'temporal_windows' in results and results['temporal_windows']:
                scores.append(results['temporal_windows'].optimization_score)
                weights.append(0.25)
            
            # Merging thresholds score
            if 'merging_thresholds' in results and results['merging_thresholds']:
                scores.append(results['merging_thresholds'].optimization_score)
                weights.append(0.2)
            
            # Validation thresholds score
            if 'validation_thresholds' in results and results['validation_thresholds']:
                scores.append(results['validation_thresholds'].optimization_score)
                weights.append(0.25)
            
            if not scores:
                return 0.0
            
            # Weighted average
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            overall_score = np.average(scores, weights=weights)
            return overall_score
            
        except Exception as e:
            logger.warning(f"Overall score calculation failed: {e}")
            return 0.0
    
    def _create_optimization_summary(self, results: Dict[str, Any], overall_score: float) -> Dict[str, Any]:
        """Create optimization summary."""
        try:
            summary = {
                'overall_score': overall_score,
                'optimization_time': time.time() - self.start_time if self.start_time else 0.0,
                'components_optimized': len(results),
                'component_results': {}
            }
            
            for component, result in results.items():
                if result:
                    summary['component_results'][component] = {
                        'score': getattr(result, 'optimization_score', 0.0),
                        'n_trials': len(getattr(result, 'optimization_history', [])),
                        'converged': getattr(result, 'convergence_info', {}).get('converged', False)
                    }
            
            return summary
            
        except Exception as e:
            logger.warning(f"Optimization summary creation failed: {e}")
            return {'overall_score': overall_score, 'error': str(e)}
    
    def _calculate_convergence_info(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate convergence information."""
        try:
            convergence_info = {
                'all_converged': True,
                'convergence_details': {}
            }
            
            for component, result in results.items():
                if result and hasattr(result, 'convergence_info'):
                    conv_info = result.convergence_info
                    convergence_info['convergence_details'][component] = {
                        'converged': conv_info.get('converged', False),
                        'n_trials': conv_info.get('n_trials', 0),
                        'best_score': conv_info.get('best_score', 0.0)
                    }
                    
                    if not conv_info.get('converged', False):
                        convergence_info['all_converged'] = False
            
            return convergence_info
            
        except Exception as e:
            logger.warning(f"Convergence info calculation failed: {e}")
            return {'all_converged': False, 'error': str(e)}
    
    def _save_optimization_results(self, result: DataDrivenClusteringResult) -> None:
        """Save optimization results to disk."""
        try:
            if not self.config.enable_caching:
                return
            
            # Create cache directory
            cache_dir = Path(self.config.cache_directory)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            result_file = cache_dir / f"optimization_results_{timestamp}.pkl"
            
            import pickle
            with open(result_file, 'wb') as f:
                pickle.dump(result, f)
            
            logger.info(f"💾 Optimization results saved to {result_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save optimization results: {e}")
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history.copy()
    
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Get optimal parameters from the last optimization."""
        # This would typically return the parameters from the last successful optimization
        # For now, return empty dict
        return {}
    
    def apply_optimal_parameters(self, 
                               features: np.ndarray,
                               market_data: pd.DataFrame,
                               clustering_func: Callable,
                               optimal_parameters: Dict[str, Any]) -> np.ndarray:
        """Apply optimal parameters to clustering."""
        try:
            # Apply feature weights if available
            if 'w_returns' in optimal_parameters:
                features = self._apply_feature_weights(features, optimal_parameters)
            
            # Apply temporal windows if available
            if 'window_size' in optimal_parameters:
                # This would typically involve windowing the data
                pass
            
            # Perform clustering
            cluster_labels = clustering_func(features)
            
            # Apply merging thresholds if available
            if 'similarity_threshold' in optimal_parameters:
                # This would typically involve merging similar clusters
                pass
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Failed to apply optimal parameters: {e}")
            raise
    
    def _apply_feature_weights(self, features: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply feature weights to features."""
        try:
            # This is a simplified version - in practice, you'd need to know
            # which features belong to which groups
            weighted_features = features.copy()
            
            # Apply weights (this is a placeholder implementation)
            if 'w_returns' in parameters:
                # Apply returns weight to returns features
                pass
            
            if 'w_vol' in parameters:
                # Apply volatility weight to volatility features
                pass
            
            if 'w_volume' in parameters:
                # Apply volume weight to volume features
                pass
            
            return weighted_features
            
        except Exception as e:
            logger.warning(f"Feature weight application failed: {e}")
            return features