"""
Enhanced Advanced Feature Selector

This module implements the enhanced advanced feature selector with
all improvements integrated.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

# Import enhanced components
from .enhanced_config import EnhancedAdvancedConfig
from .enhanced_ensemble_selector import EnhancedEnsembleAdvancedSelector

logger = logging.getLogger(__name__)

class EnhancedAdvancedFeatureSelector:
    """Enhanced advanced feature selector with all improvements."""
    
    def __init__(self, config: Optional[EnhancedAdvancedConfig] = None):
        """Initialize enhanced advanced selector."""
        self.config = config or EnhancedAdvancedConfig()
        self.logger = logger.getChild('EnhancedAdvancedFeatureSelector')
        
        # Initialize hardware optimization
        if self.config.enable_hardware_optimization:
            self.cpu_optimizer = M1CPUOptimizer()
            hw_config = HardwareConfig(
                cpu_optimization_level='aggressive',
                enable_adaptive_optimization=True
            )
            self.hardware_manager = UnifiedHardwareManager(hw_config)
        else:
            self.cpu_optimizer = None
            self.hardware_manager = None
        
        # Initialize ensemble selector
        self.ensemble_selector = EnhancedEnsembleAdvancedSelector(self.config.ensemble_config)
        
        # Performance tracking
        self.performance_stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'method_selections': {},
            'avg_selection_time': 0.0,
            'error_recoveries': 0
        }
        
        tprint_success("🔧 EnhancedAdvancedFeatureSelector initialized")
    
    def select_features(self, X: np.ndarray, y: np.ndarray,
                       method: str = 'enhanced_ensemble',
                       target_features: Optional[int] = None,
                       target_percentage: Optional[float] = None,
                       target_performance_threshold: Optional[float] = None,
                       feature_names: Optional[List[str]] = None,
                       **kwargs) -> Dict[str, Any]:
        """Select features using enhanced methods."""
        tprint_info(f"🔧 Enhanced selection: {method}")
        
        start_time = time.time()
        
        try:
            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Select method
            if method == 'auto':
                method = self._select_optimal_method(X, y)
                tprint_debug(f"🔧 Auto-selected method: {method}")
            
            # Run selection with error recovery
            result = self._run_selection_with_recovery(
                X, y, method, target_features, target_percentage, 
                target_performance_threshold, feature_names, **kwargs
            )
            
            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time
            self.performance_stats['total_selections'] += 1
            
            if result.get('success', False):
                self.performance_stats['successful_selections'] += 1
                self.performance_stats['method_selections'][method] = self.performance_stats['method_selections'].get(method, 0) + 1
            
            self.performance_stats['avg_selection_time'] = (
                (self.performance_stats['avg_selection_time'] * (self.performance_stats['total_selections'] - 1) + 
                 execution_time) / self.performance_stats['total_selections']
            )
            
            result['execution_time'] = execution_time
            result['method_used'] = method
            
            tprint_success(f"✅ Enhanced selection completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced selection failed: {e}")
            end_time = time.time()
            return {
                'success': False,
                'error': str(e),
                'execution_time': end_time - start_time,
                'method_used': method
            }
    
    def _select_optimal_method(self, X: np.ndarray, y: np.ndarray) -> str:
        """Select optimal method based on data characteristics."""
        if not self.config.enable_auto_method_selection:
            return 'enhanced_ensemble'
        
        tprint_debug("🔧 Selecting optimal method")
        
        try:
            # Analyze data characteristics
            data_characteristics = self._analyze_data_characteristics(X, y)
            
            # Select method based on characteristics
            if data_characteristics['is_high_dimensional'] and data_characteristics['is_sparse']:
                method = 'enhanced_ensemble'  # Best for high-dim sparse data
            elif data_characteristics['is_small_sample']:
                method = 'enhanced_ensemble'  # Best for small samples
            elif data_characteristics['is_time_series']:
                method = 'enhanced_ensemble'  # Best for time series
            else:
                method = 'enhanced_ensemble'  # Default to ensemble
            
            tprint_debug(f"🔧 Selected method: {method} based on data characteristics")
            return method
            
        except Exception as e:
            self.logger.warning(f"Optimal method selection failed: {e}")
            return 'enhanced_ensemble'
    
    def _analyze_data_characteristics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze data characteristics to inform method selection."""
        try:
            n_samples, n_features = X.shape
            
            # High dimensional check
            is_high_dimensional = n_features > n_samples * 2
            
            # Sparse data check
            is_sparse = np.sum(X == 0) / X.size > 0.5
            
            # Small sample check
            is_small_sample = n_samples < 100
            
            # Time series check (simplified)
            is_time_series = len(np.unique(y)) > 10 and np.std(np.diff(y)) < np.std(y) * 0.1
            
            # Categorical data check
            is_categorical = np.all(X == X.astype(int)) and np.max(X) < 10
            
            return {
                'n_samples': n_samples,
                'n_features': n_features,
                'is_high_dimensional': is_high_dimensional,
                'is_sparse': is_sparse,
                'is_small_sample': is_small_sample,
                'is_time_series': is_time_series,
                'is_categorical': is_categorical,
                'feature_ratio': n_features / n_samples
            }
            
        except Exception as e:
            self.logger.warning(f"Data characteristics analysis failed: {e}")
            return {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'is_high_dimensional': False,
                'is_sparse': False,
                'is_small_sample': False,
                'is_time_series': False,
                'is_categorical': False,
                'feature_ratio': X.shape[1] / X.shape[0]
            }
    
    def _run_selection_with_recovery(self, X: np.ndarray, y: np.ndarray,
                                   method: str, target_features: Optional[int],
                                   target_percentage: Optional[float],
                                   target_performance_threshold: Optional[float],
                                   feature_names: List[str], **kwargs) -> Dict[str, Any]:
        """Run selection with error recovery."""
        max_attempts = self.config.max_retry_attempts if self.config.enable_error_recovery else 1
        
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    tprint_warning(f"⚠️ Retry attempt {attempt + 1}/{max_attempts}")
                    time.sleep(self.config.retry_delay * attempt)
                
                # Run selection
                result = self.ensemble_selector.select_features(
                    X, y, target_features, target_percentage, 
                    target_performance_threshold, feature_names, **kwargs
                )
                
                if result.get('success', False):
                    return result
                else:
                    if attempt < max_attempts - 1:
                        tprint_warning(f"⚠️ Selection failed, retrying: {result.get('error', 'Unknown error')}")
                        continue
                    else:
                        return result
                        
            except Exception as e:
                if attempt < max_attempts - 1:
                    tprint_warning(f"⚠️ Selection exception, retrying: {e}")
                    continue
                else:
                    return {
                        'success': False,
                        'error': str(e),
                        'attempts': attempt + 1
                    }
        
        # If we get here, all attempts failed
        return {
            'success': False,
            'error': 'All retry attempts failed',
            'attempts': max_attempts
        }
    
    def compare_methods(self, X: np.ndarray, y: np.ndarray,
                       target_features: Optional[int] = None,
                       target_percentage: Optional[float] = None,
                       target_performance_threshold: Optional[float] = None,
                       **kwargs) -> Dict[str, Any]:
        """Compare all enhanced methods."""
        tprint_info("🔧 Comparing enhanced methods")
        
        try:
            # Run ensemble method
            ensemble_result = self.select_features(
                X, y, 'enhanced_ensemble', target_features, 
                target_percentage, target_performance_threshold, **kwargs
            )
            
            # Get individual method results from ensemble
            individual_results = ensemble_result.get('individual_results', {})
            
            # Combine results
            all_results = individual_results.copy()
            all_results['enhanced_ensemble'] = ensemble_result
            
            # Create comparison summary
            comparison_summary = self._create_comparison_summary(all_results)
            
            # Add performance insights
            performance_insights = self._create_performance_insights(all_results)
            
            return {
                'success': True,
                'results': all_results,
                'comparison_summary': comparison_summary,
                'performance_insights': performance_insights,
                'data_characteristics': self._analyze_data_characteristics(X, y)
            }
            
        except Exception as e:
            self.logger.error(f"Method comparison failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison summary for all methods."""
        summary = {}
        
        for method_name, result in results.items():
            if result.get('success', False):
                summary[method_name] = {
                    'n_selected': result.get('n_selected', 0),
                    'n_total': result.get('n_total', 0),
                    'selection_ratio': result.get('n_selected', 0) / max(1, result.get('n_total', 1)),
                    'execution_time': result.get('execution_time', 0.0),
                    'method_type': 'individual' if method_name != 'enhanced_ensemble' else 'ensemble',
                    'success': True
                }
                
                # Add validation metrics
                validation_results = result.get('validation_results', {})
                if validation_results.get('cross_validation', {}).get('success', False):
                    cv_results = validation_results['cross_validation']
                    aggregated = cv_results.get('aggregated_results', {})
                    if 'r2_mean' in aggregated:
                        summary[method_name]['cv_r2'] = aggregated['r2_mean']
                    elif 'accuracy_mean' in aggregated:
                        summary[method_name]['cv_accuracy'] = aggregated['accuracy_mean']
                
                # Add confidence metrics
                confidence_scores = result.get('confidence_scores', {})
                if confidence_scores:
                    confidences = [scores['confidence_score'] for scores in confidence_scores.values()]
                    if confidences:
                        summary[method_name]['avg_confidence'] = np.mean(confidences)
                        summary[method_name]['high_confidence_count'] = sum(1 for c in confidences if c >= 0.8)
                        summary[method_name]['consensus_count'] = sum(1 for c in confidences if c >= 0.5)
            else:
                summary[method_name] = {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'method_type': 'individual' if method_name != 'enhanced_ensemble' else 'ensemble'
                }
        
        return summary
    
    def _create_performance_insights(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create performance insights from comparison results."""
        insights = {
            'best_method': None,
            'fastest_method': None,
            'most_consistent_method': None,
            'performance_ranking': [],
            'confidence_ranking': []
        }
        
        try:
            # Find best performing method
            method_scores = {}
            method_times = {}
            method_confidences = {}
            
            for method_name, result in results.items():
                if result.get('success', False):
                    # Performance score
                    validation_results = result.get('validation_results', {})
                    cv_results = validation_results.get('cross_validation', {})
                    if cv_results.get('success', False):
                        aggregated = cv_results.get('aggregated_results', {})
                        if 'r2_mean' in aggregated:
                            method_scores[method_name] = aggregated['r2_mean']
                        elif 'accuracy_mean' in aggregated:
                            method_scores[method_name] = aggregated['accuracy_mean']
                    
                    # Execution time
                    method_times[method_name] = result.get('execution_time', float('inf'))
                    
                    # Confidence score
                    confidence_scores = result.get('confidence_scores', {})
                    if confidence_scores:
                        confidences = [scores['confidence_score'] for scores in confidence_scores.values()]
                        method_confidences[method_name] = np.mean(confidences)
            
            # Rank methods
            if method_scores:
                insights['best_method'] = max(method_scores.items(), key=lambda x: x[1])[0]
                insights['performance_ranking'] = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
            
            if method_times:
                insights['fastest_method'] = min(method_times.items(), key=lambda x: x[1])[0]
            
            if method_confidences:
                insights['most_consistent_method'] = max(method_confidences.items(), key=lambda x: x[1])[0]
                insights['confidence_ranking'] = sorted(method_confidences.items(), key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Performance insights creation failed: {e}")
        
        return insights
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add ensemble statistics
        stats['ensemble_statistics'] = self.ensemble_selector.get_ensemble_statistics()
        
        # Calculate success rate
        stats['success_rate'] = self.performance_stats['successful_selections'] / max(1, self.performance_stats['total_selections'])
        
        # Add method usage statistics
        total_method_selections = sum(stats['method_selections'].values())
        if total_method_selections > 0:
            stats['method_usage_ratio'] = {
                method: count / total_method_selections 
                for method, count in stats['method_selections'].items()
            }
        
        return stats
    
    def get_enhanced_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about enhanced selection."""
        insights = {
            'total_selections': self.performance_stats['total_selections'],
            'success_rate': self.performance_stats['successful_selections'] / max(1, self.performance_stats['total_selections']),
            'avg_selection_time': self.performance_stats['avg_selection_time'],
            'error_recoveries': self.performance_stats['error_recoveries'],
            'method_usage': self.performance_stats['method_selections'],
            'ensemble_insights': self.ensemble_selector.get_ensemble_insights()
        }
        
        return insights
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.performance_stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'method_selections': {},
            'avg_selection_time': 0.0,
            'error_recoveries': 0
        }
        
        # Reset ensemble statistics
        self.ensemble_selector.adaptive_weighting.reset_weights(list(self.ensemble_selector.selectors.keys()))
        
        tprint_info("🔧 Statistics reset")
    
    def get_method_recommendations(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Get method recommendations based on data characteristics."""
        tprint_info("🔧 Generating method recommendations")
        
        try:
            # Analyze data characteristics
            characteristics = self._analyze_data_characteristics(X, y)
            
            # Generate recommendations
            recommendations = {
                'data_characteristics': characteristics,
                'recommended_method': 'enhanced_ensemble',
                'reasoning': [],
                'alternative_methods': [],
                'configuration_suggestions': {}
            }
            
            # Add reasoning based on characteristics
            if characteristics['is_high_dimensional']:
                recommendations['reasoning'].append("High-dimensional data: Enhanced ensemble recommended for robust selection")
            
            if characteristics['is_sparse']:
                recommendations['reasoning'].append("Sparse data: Enhanced ensemble handles sparse data well")
            
            if characteristics['is_small_sample']:
                recommendations['reasoning'].append("Small sample size: Enhanced ensemble with cross-validation recommended")
            
            if characteristics['is_time_series']:
                recommendations['reasoning'].append("Time series data: Enhanced ensemble with temporal validation recommended")
            
            if characteristics['is_categorical']:
                recommendations['reasoning'].append("Categorical data: Enhanced ensemble with appropriate encoding recommended")
            
            # Add configuration suggestions
            if characteristics['is_high_dimensional']:
                recommendations['configuration_suggestions']['enable_pre_filtering'] = True
                recommendations['configuration_suggestions']['target_percentage'] = 0.1
            
            if characteristics['is_small_sample']:
                recommendations['configuration_suggestions']['cv_folds'] = 3
                recommendations['configuration_suggestions']['enable_stability_metrics'] = True
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Method recommendations failed: {e}")
            return {
                'data_characteristics': {},
                'recommended_method': 'enhanced_ensemble',
                'reasoning': ['Error in analysis'],
                'alternative_methods': [],
                'configuration_suggestions': {}
            }