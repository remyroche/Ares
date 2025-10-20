"""
Enhanced Ensemble Advanced Selector

This module implements the enhanced ensemble advanced selector with
adaptive weighting, confidence scoring, native validation, and dynamic
feature selection.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware import (
    get_integrated_hardware_manager,
    memory_efficient,
    performance_tracked,
    smart_cache,
    auto_optimize,
    WorkloadType,
    OptimizationLevel
)

# Import enhanced components
from .enhanced_config import EnhancedEnsembleConfig
from .adaptive_weighting import AdaptiveWeightingSystem
from .confidence_scoring import ConfidenceScoringSystem
from .native_validation import NativeValidationFramework
from .dynamic_selection import DynamicFeatureSelector

# Import base selectors
from .advanced_selector import (
    LASSOFeatureSelector,
    RandomForestFeatureSelector,
    LightGBMFeatureSelector
)

logger = logging.getLogger(__name__)

class EnhancedEnsembleAdvancedSelector:
    """Enhanced ensemble advanced selector with all improvements."""

    def __init__(self, config: Optional[EnhancedEnsembleConfig] = None):
        """Initialize enhanced ensemble selector."""
        self.config = config or EnhancedEnsembleConfig()
        self.logger = logger.getChild('EnhancedEnsembleAdvancedSelector')

        # Initialize hardware optimization
        if self.config.enable_hardware_optimization:
            self.hardware_manager = get_integrated_hardware_manager()
        else:
            self.hardware_manager = None

        # Initialize enhanced components
        self.adaptive_weighting = AdaptiveWeightingSystem(
            self.config.adaptive_weighting, self.hardware_manager
        )
        self.confidence_scoring = ConfidenceScoringSystem(
            self.config.confidence_scoring, self.hardware_manager
        )
        self.native_validation = NativeValidationFramework(
            self.config.native_validation, self.hardware_manager
        )
        self.dynamic_selection = DynamicFeatureSelector(
            self.config.dynamic_selection, self.hardware_manager
        )

        # Initialize individual selectors
        self.selectors = {
            'lasso': LASSOFeatureSelector(),
            'random_forest': RandomForestFeatureSelector(),
            'lightgbm': LightGBMFeatureSelector() if self._is_lightgbm_available() else None
        }

        # Remove None selectors
        self.selectors = {k: v for k, v in self.selectors.items() if v is not None}

        # Initialize adaptive weights
        self.adaptive_weighting.initialize_weights(list(self.selectors.keys()))

        # Performance tracking
        self.performance_stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'avg_selection_time': 0.0,
            'weight_updates': 0,
            'confidence_calculations': 0
        }

        tprint_success(f"🔧 EnhancedEnsembleAdvancedSelector initialized with {len(self.selectors)} methods")

    def _is_lightgbm_available(self) -> bool:
        """Check if LightGBM is available."""
        try:
            import lightgbm
            return True
        except ImportError:
            return False

    @memory_efficient(memory_threshold_mb=1000.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
    @smart_cache(ttl=1800)  # Cache for 30 minutes
    def select_features(self, X: np.ndarray, y: np.ndarray,
                       target_features: Optional[int] = None,
                       target_percentage: Optional[float] = None,
                       target_performance_threshold: Optional[float] = None,
                       feature_names: Optional[List[str]] = None,
                       **kwargs) -> Dict[str, Any]:
        """Select features using enhanced ensemble method."""
        tprint_info(f"🔧 Enhanced ensemble selection: {X.shape}")

        start_time = time.time()

        try:
            # Optimize hardware for ensemble workload
            if self.hardware_manager:
                self.hardware_manager.optimize_for_workload(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE)
                # Pre-optimize data
                X = self.hardware_manager.process_data_with_optimization(X, WorkloadType.ML_TRAINING)
                y = self.hardware_manager.process_data_with_optimization(y, WorkloadType.ML_TRAINING)
            
            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Step 1: Determine target count
            target_info = self.dynamic_selection.determine_target_count(
                X, y, target_features, target_percentage, target_performance_threshold
            )
            target_count = target_info['target_count']

            tprint_debug(f"🔧 Target count determined: {target_count} features")

            # Step 2: Run individual methods
            individual_results = self._run_individual_methods(X, y, target_count, feature_names, **kwargs)

            # Step 3: Native validation
            validation_results = self.native_validation.validate_selection_methods(
                X, y, individual_results, feature_names
            )

            # Step 4: Update adaptive weights based on performance
            if validation_results.get('cross_validation', {}).get('success', False):
                method_performances = self._extract_method_performances(validation_results)
                self.adaptive_weighting.update_weights(method_performances)
                self.performance_stats['weight_updates'] += 1

            # Step 5: Calculate confidence scores
            confidence_scores = self.confidence_scoring.calculate_confidence_scores(
                individual_results, feature_names
            )
            self.performance_stats['confidence_calculations'] += 1

            # Step 6: Create ensemble selection
            ensemble_result = self._create_ensemble_selection(
                individual_results, confidence_scores, target_count, feature_names
            )

            # Step 7: Final validation
            final_validation = self._validate_final_selection(X, y, ensemble_result, feature_names)

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time
            self.performance_stats['total_selections'] += 1
            self.performance_stats['successful_selections'] += 1
            self.performance_stats['avg_selection_time'] = (
                (self.performance_stats['avg_selection_time'] * (self.performance_stats['total_selections'] - 1) +
                 execution_time) / self.performance_stats['total_selections']
            )

            result = {
                'success': True,
                'selected_features': ensemble_result['selected_features'],
                'selected_indices': ensemble_result['selected_indices'],
                'feature_scores': ensemble_result['feature_scores'],
                'confidence_scores': confidence_scores,
                'target_info': target_info,
                'individual_results': individual_results,
                'validation_results': validation_results,
                'final_validation': final_validation,
                'adaptive_weights': self.adaptive_weighting.get_current_weights(),
                'n_selected': len(ensemble_result['selected_features']),
                'n_total': X.shape[1],
                'method': 'enhanced_ensemble',
                'execution_time': execution_time
            }

            tprint_success(f"✅ Enhanced ensemble selection completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Enhanced ensemble selection failed: {e}")
            end_time = time.time()
            return {
                'success': False,
                'error': str(e),
                'execution_time': end_time - start_time
            }

    def _run_individual_methods(self, X: np.ndarray, y: np.ndarray,
                              target_count: int, feature_names: List[str],
                              **kwargs) -> Dict[str, Dict[str, Any]]:
        """Run individual selection methods."""
        tprint_debug("🔧 Running individual methods")

        individual_results = {}

        for method_name, selector in self.selectors.items():
            try:
                tprint_debug(f"🔧 Running {method_name}")
                result = selector.select_features(X, y, n_features=target_count, **kwargs)
                individual_results[method_name] = result
            except Exception as e:
                self.logger.warning(f"Method {method_name} failed: {e}")
                individual_results[method_name] = {
                    'success': False,
                    'error': str(e),
                    'selected_features': [],
                    'selected_indices': []
                }

        return individual_results

    def _extract_method_performances(self, validation_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract method performances from validation results."""
        method_performances = {}

        cv_results = validation_results.get('cross_validation', {})
        if cv_results.get('success', False):
            method_results = cv_results.get('method_results', {})
            for method_name, result in method_results.items():
                if result.get('success', False):
                    aggregated_metrics = result.get('aggregated_metrics', {})
                    # Use R2 score if available, otherwise use accuracy
                    if 'r2_mean' in aggregated_metrics:
                        method_performances[method_name] = aggregated_metrics['r2_mean']
                    elif 'accuracy_mean' in aggregated_metrics:
                        method_performances[method_name] = aggregated_metrics['accuracy_mean']
                    else:
                        method_performances[method_name] = 0.5  # Default performance

        return method_performances

    def _create_ensemble_selection(self, individual_results: Dict[str, Dict[str, Any]],
                                 confidence_scores: Dict[str, Dict[str, Any]],
                                 target_count: int, feature_names: List[str]) -> Dict[str, Any]:
        """Create ensemble selection from individual results."""
        tprint_debug("🔧 Creating ensemble selection")

        try:
            # Get current adaptive weights
            adaptive_weights = self.adaptive_weighting.get_current_weights()

            # Collect all selected features with their scores
            feature_scores = {}
            feature_methods = {}

            for method_name, result in individual_results.items():
                if result.get('success', False):
                    selected_features = result.get('selected_features', [])
                    feature_scores_method = result.get('feature_scores', {})
                    weight = adaptive_weights.get(method_name, 1.0)

                    for feature in selected_features:
                        if feature not in feature_scores:
                            feature_scores[feature] = 0.0
                            feature_methods[feature] = []

                        # Weighted score
                        feature_score = feature_scores_method.get(feature, 0.0)
                        feature_scores[feature] += feature_score * weight
                        feature_methods[feature].append(method_name)

            # Apply confidence scoring
            for feature, scores in confidence_scores.items():
                if feature in feature_scores:
                    confidence_weight = scores['confidence_score']
                    feature_scores[feature] *= confidence_weight

            # Sort features by combined score
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

            # Select top features
            selected_features = [feature for feature, score in sorted_features[:target_count]]
            selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]

            # Create final feature scores
            final_feature_scores = {
                feature: {
                    'combined_score': float(score),
                    'confidence_score': confidence_scores.get(feature, {}).get('confidence_score', 0.0),
                    'method_agreement': len(feature_methods.get(feature, [])),
                    'methods': feature_methods.get(feature, [])
                }
                for feature, score in sorted_features[:target_count]
            }

            return {
                'selected_features': selected_features,
                'selected_indices': selected_indices,
                'feature_scores': final_feature_scores,
                'n_selected': len(selected_features)
            }

        except Exception as e:
            self.logger.error(f"Ensemble selection creation failed: {e}")
            return {
                'selected_features': [],
                'selected_indices': [],
                'feature_scores': {},
                'n_selected': 0
            }

    def _validate_final_selection(self, X: np.ndarray, y: np.ndarray,
                                ensemble_result: Dict[str, Any],
                                feature_names: List[str]) -> Dict[str, Any]:
        """Validate final ensemble selection."""
        tprint_debug("🔧 Validating final selection")

        try:
            selected_features = ensemble_result['selected_features']
            selected_indices = ensemble_result['selected_indices']

            if not selected_features:
                return {'success': False, 'error': 'No features selected'}

            # Get selected features data
            X_selected = X[:, selected_indices]

            # Calculate performance metrics
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score

            model = LinearRegression()
            model.fit(X_selected, y)
            y_pred = model.predict(X_selected)
            r2 = r2_score(y, y_pred)

            # Calculate feature quality metrics
            feature_quality = self._calculate_feature_quality(X_selected, y)

            return {
                'success': True,
                'r2_score': float(r2),
                'feature_quality': feature_quality,
                'n_selected': len(selected_features),
                'selection_ratio': len(selected_features) / len(feature_names)
            }

        except Exception as e:
            self.logger.warning(f"Final selection validation failed: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_feature_quality(self, X_selected: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate quality metrics for selected features."""
        try:
            # Variance metrics
            feature_variances = np.var(X_selected, axis=0)

            # Correlation metrics
            if X_selected.shape[1] > 1:
                corr_matrix = np.corrcoef(X_selected.T)
                high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.95) - X_selected.shape[1]
            else:
                high_corr_pairs = 0

            return {
                'mean_variance': float(np.mean(feature_variances)),
                'variance_std': float(np.std(feature_variances)),
                'high_correlation_pairs': int(high_corr_pairs),
                'n_features': X_selected.shape[1]
            }

        except Exception as e:
            self.logger.warning(f"Feature quality calculation failed: {e}")
            return {}

    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        stats = self.performance_stats.copy()

        # Add component statistics
        stats['adaptive_weighting'] = self.adaptive_weighting.get_performance_stats()
        stats['confidence_scoring'] = self.confidence_scoring.get_confidence_statistics()
        stats['native_validation'] = self.native_validation.get_validation_statistics()
        stats['dynamic_selection'] = self.dynamic_selection.get_selection_statistics()

        return stats

    def get_ensemble_insights(self) -> Dict[str, Any]:
        """Get insights about ensemble behavior."""
        insights = {
            'total_selections': self.performance_stats['total_selections'],
            'success_rate': self.performance_stats['successful_selections'] / max(1, self.performance_stats['total_selections']),
            'avg_selection_time': self.performance_stats['avg_selection_time'],
            'component_insights': {
                'adaptive_weighting': self.adaptive_weighting.get_weight_insights(),
                'confidence_scoring': self.confidence_scoring.get_confidence_insights({}),
                'native_validation': self.native_validation.get_validation_insights(),
                'dynamic_selection': self.dynamic_selection.get_selection_insights()
            }
        }

        return insights

    def compare_methods(self, X: np.ndarray, y: np.ndarray,
                       target_features: Optional[int] = None,
                       target_percentage: Optional[float] = None,
                       **kwargs) -> Dict[str, Any]:
        """Compare all methods including ensemble."""
        tprint_info("🔧 Comparing enhanced methods")

        try:
            # Determine target count
            target_info = self.dynamic_selection.determine_target_count(
                X, y, target_features, target_percentage
            )
            target_count = target_info['target_count']

            # Run individual methods
            individual_results = self._run_individual_methods(X, y, target_count, **kwargs)

            # Run ensemble method
            ensemble_result = self.select_features(X, y, target_features, target_percentage, **kwargs)

            # Combine results
            all_results = individual_results.copy()
            all_results['enhanced_ensemble'] = ensemble_result

            # Create comparison summary
            comparison_summary = self._create_comparison_summary(all_results)

            return {
                'success': True,
                'results': all_results,
                'comparison_summary': comparison_summary,
                'target_info': target_info
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
                    'method_type': 'individual' if method_name != 'enhanced_ensemble' else 'ensemble'
                }

                # Add method-specific metrics
                if 'r2_score' in result.get('final_validation', {}):
                    summary[method_name]['r2_score'] = result['final_validation']['r2_score']

                if 'confidence_scores' in result:
                    confidences = [scores['confidence_score'] for scores in result['confidence_scores'].values()]
                    if confidences:
                        summary[method_name]['avg_confidence'] = np.mean(confidences)
                        summary[method_name]['high_confidence_count'] = sum(1 for c in confidences if c >= 0.8)

        return summary
