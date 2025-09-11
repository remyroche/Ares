from src.utils.tprint import tprint

"""
Main Feature Selection Framework

This module provides the main orchestrator that combines all feature selection
components into a comprehensive, modular framework.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
from datetime import datetime
import time
import warnings

# Import all components
from .base_framework import BaseFeatureSelectionFramework
from .data_validation import DataValidator
from .selection_methods import (
    MRMRSelector, LassoStabilitySelector, CorrelationBasedFilter,
    RecursiveFeatureEliminator, FeatureImportanceRanker
)
from .stability_analysis import StabilityAnalyzer
from .performance_monitoring import PerformanceMonitor, MemoryOptimizer
from .quality_metrics import QualityMetricsCalculator
from .temporal_analysis import TemporalAnalyzer
from .causal_analysis import CausalAnalyzer

# Enhanced dependency management
try:
    from ...utils.logger import get_logger
    _LOGGER = get_logger("FeatureSelection.MainFramework")
    tprint("✅ Custom logger available for FeatureSelection.MainFramework")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("FeatureSelection.MainFramework")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER


class FeatureSelectionFramework(BaseFeatureSelectionFramework):
    """Main feature selection framework orchestrating all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the main feature selection framework."""
        super().__init__(config)
        self.logger = logger.getChild('MainFramework')
        
        _LOGGER.info("🚀 Initializing Main FeatureSelectionFramework...")
        
        # Initialize all components
        self._initialize_components()
        
        _LOGGER.info("✅ Main FeatureSelectionFramework initialized successfully")

    def _initialize_components(self):
        """Initialize all feature selection components."""
        try:
            # Data validation component
            self.data_validator = DataValidator(self.config.get('data_validation', {}))
            _LOGGER.info("✅ DataValidator initialized")
            
            # Selection methods
            self.mrmr_selector = MRMRSelector(self.config.get('mrmr', {}))
            self.lasso_stability_selector = LassoStabilitySelector(self.config.get('lasso_stability', {}))
            self.correlation_filter = CorrelationBasedFilter(self.config.get('correlation_filter', {}))
            self.rfe_selector = RecursiveFeatureEliminator(self.config.get('rfe', {}))
            self.importance_ranker = FeatureImportanceRanker(self.config.get('importance', {}))
            _LOGGER.info("✅ Selection methods initialized")
            
            # Analysis components
            self.stability_analyzer = StabilityAnalyzer(self.config.get('stability_analysis', {}))
            self.quality_calculator = QualityMetricsCalculator(self.config.get('quality_metrics', {}))
            self.temporal_analyzer = TemporalAnalyzer(self.config.get('temporal_analysis', {}))
            self.causal_analyzer = CausalAnalyzer(self.config.get('causal_analysis', {}))
            _LOGGER.info("✅ Analysis components initialized")
            
            # Performance monitoring
            self.performance_monitor = PerformanceMonitor(max_history=1000)
            self.memory_optimizer = MemoryOptimizer(self.config.get('memory_optimization', {}))
            _LOGGER.info("✅ Performance monitoring initialized")
            
        except Exception as e:
            _LOGGER.error(f"❌ Component initialization failed: {e}")
            raise

    def run_comprehensive_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                          feature_names: List[str],
                                          target_features: Optional[int] = None,
                                          model_type: str = 'default',
                                          enable_stability_analysis: bool = True,
                                          enable_temporal_analysis: bool = False,
                                          enable_causal_analysis: bool = False) -> Dict[str, Any]:
        """Run comprehensive feature selection pipeline."""
        start_time = time.time()
        _LOGGER.info(f"🚀 Starting comprehensive feature selection pipeline...")
        _LOGGER.info(f"📊 Parameters - Data shape: {X.shape}, Target features: {target_features}")
        
        try:
            # Step 1: Data validation and cleaning
            _LOGGER.info("🔍 Step 1: Data validation and cleaning...")
            validation_result = self.data_validator.validate_data_quality(X, y)
            
            if not validation_result.get('is_valid', True):
                _LOGGER.warning(f"⚠️ Data validation issues: {validation_result.get('issues', [])}")
            
            # Clean data if needed
            X_cleaned, y_cleaned, cleaning_log = self.data_validator.clean_data(
                X, y, remove_constant=True, remove_high_corr=True, remove_nan_inf=True
            )
            
            _LOGGER.info(f"🧹 Data cleaning: {cleaning_log.get('features_removed_count', 0)} features removed")
            
            # Step 2: Determine target feature count
            if target_features is None:
                target_features = self.get_model_target_features(model_type)
            
            _LOGGER.info(f"🎯 Target features: {target_features}")
            
            # Step 3: Apply selection methods
            _LOGGER.info("🔍 Step 3: Applying feature selection methods...")
            selection_results = self._apply_selection_methods(
                X_cleaned, y_cleaned, feature_names, target_features
            )
            
            # Step 4: Stability analysis (if enabled)
            stability_results = {}
            if enable_stability_analysis:
                _LOGGER.info("📈 Step 4: Stability analysis...")
                stability_results = self._perform_stability_analysis(
                    X_cleaned, y_cleaned, feature_names, selection_results
                )
            
            # Step 5: Temporal analysis (if enabled)
            temporal_results = {}
            if enable_temporal_analysis:
                _LOGGER.info("⏰ Step 5: Temporal analysis...")
                temporal_results = self._perform_temporal_analysis(
                    X_cleaned, y_cleaned, feature_names
                )
            
            # Step 6: Causal analysis (if enabled)
            causal_results = {}
            if enable_causal_analysis:
                _LOGGER.info("🔗 Step 6: Causal analysis...")
                causal_results = self._perform_causal_analysis(
                    X_cleaned, y_cleaned, feature_names
                )
            
            # Step 7: Quality assessment
            _LOGGER.info("📊 Step 7: Quality assessment...")
            quality_results = self._assess_selection_quality(
                X_cleaned, y_cleaned, feature_names, selection_results,
                stability_results, temporal_results, causal_results
            )
            
            # Step 8: Final feature selection
            _LOGGER.info("🎯 Step 8: Final feature selection...")
            final_selection = self._select_final_features(
                selection_results, stability_results, quality_results
            )
            
            execution_time = time.time() - start_time
            
            # Compile comprehensive results
            result = {
                'final_selected_features': final_selection['selected_features'],
                'final_selected_indices': final_selection['selected_indices'],
                'selection_results': selection_results,
                'stability_results': stability_results,
                'temporal_results': temporal_results,
                'causal_results': causal_results,
                'quality_results': quality_results,
                'validation_result': validation_result,
                'cleaning_log': cleaning_log,
                'pipeline_summary': {
                    'execution_time': execution_time,
                    'target_features': target_features,
                    'final_feature_count': len(final_selection['selected_features']),
                    'data_shape_original': X.shape,
                    'data_shape_cleaned': X_cleaned.shape,
                    'features_removed': cleaning_log.get('features_removed_count', 0),
                    'model_type': model_type,
                    'stability_analysis_enabled': enable_stability_analysis,
                    'temporal_analysis_enabled': enable_temporal_analysis,
                    'causal_analysis_enabled': enable_causal_analysis
                },
                'success': True
            }
            
            _LOGGER.info(f"✅ Comprehensive feature selection completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Final selection: {len(final_selection['selected_features'])} features")
            _LOGGER.info(f"🎯 Selected features: {final_selection['selected_features']}")
            
            return result
            
        except Exception as e:
            _LOGGER.error(f"❌ Comprehensive feature selection failed: {e}")
            return {
                'final_selected_features': [],
                'final_selected_indices': [],
                'selection_results': {},
                'stability_results': {},
                'temporal_results': {},
                'causal_results': {},
                'quality_results': {},
                'pipeline_summary': {
                    'execution_time': time.time() - start_time,
                    'error': str(e)
                },
                'success': False
            }

    def _apply_selection_methods(self, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str], target_features: int) -> Dict[str, Any]:
        """Apply all feature selection methods."""
        try:
            selection_results = {}
            
            # mRMR selection
            _LOGGER.info("🔍 Applying mRMR selection...")
            mrmr_result = self.mrmr_selector.select_features(X, y, feature_names, target_features)
            selection_results['mrmr'] = mrmr_result
            
            # LASSO stability selection
            _LOGGER.info("🔍 Applying LASSO stability selection...")
            lasso_result = self.lasso_stability_selector.select_features(X, y, feature_names)
            selection_results['lasso_stability'] = lasso_result
            
            # Correlation-based filtering
            _LOGGER.info("🔍 Applying correlation-based filtering...")
            corr_result = self.correlation_filter.select_features(X, y, feature_names)
            selection_results['correlation_filter'] = corr_result
            
            # Feature importance ranking
            _LOGGER.info("🔍 Applying feature importance ranking...")
            importance_result = self.importance_ranker.select_features(X, y, feature_names, target_features)
            selection_results['feature_importance'] = importance_result
            
            # RFE (if sklearn available)
            try:
                _LOGGER.info("🔍 Applying recursive feature elimination...")
                rfe_result = self.rfe_selector.select_features(X, y, feature_names, target_features)
                selection_results['rfe'] = rfe_result
            except Exception as e:
                _LOGGER.warning(f"⚠️ RFE selection failed: {e}")
                selection_results['rfe'] = {'error': str(e), 'success': False}
            
            _LOGGER.info(f"✅ Applied {len(selection_results)} selection methods")
            return selection_results
            
        except Exception as e:
            _LOGGER.error(f"❌ Selection methods application failed: {e}")
            return {'error': str(e)}

    def _perform_stability_analysis(self, X: np.ndarray, y: np.ndarray, 
                                  feature_names: List[str], 
                                  selection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stability analysis."""
        try:
            stability_results = {}
            
            # Bootstrap stability analysis for each successful method
            for method_name, result in selection_results.items():
                if result.get('success', False):
                    _LOGGER.info(f"📈 Analyzing stability for {method_name}...")
                    
                    # Create selection method wrapper
                    def selection_wrapper(X_sub, y_sub, feature_names_sub, **kwargs):
                        if method_name == 'mrmr':
                            return self.mrmr_selector.select_features(X_sub, y_sub, feature_names_sub, kwargs.get('n_features', 50))
                        elif method_name == 'lasso_stability':
                            return self.lasso_stability_selector.select_features(X_sub, y_sub, feature_names_sub)
                        elif method_name == 'correlation_filter':
                            return self.correlation_filter.select_features(X_sub, y_sub, feature_names_sub)
                        elif method_name == 'feature_importance':
                            return self.importance_ranker.select_features(X_sub, y_sub, feature_names_sub, kwargs.get('n_features', 50))
                        else:
                            return {'selected_features': [], 'success': False}
                    
                    # Perform bootstrap stability analysis
                    bootstrap_result = self.stability_analyzer.analyze_bootstrap_stability(
                        X, y, feature_names, selection_wrapper, {'n_features': 50}
                    )
                    
                    stability_results[method_name] = bootstrap_result
            
            _LOGGER.info(f"✅ Stability analysis completed for {len(stability_results)} methods")
            return stability_results
            
        except Exception as e:
            _LOGGER.error(f"❌ Stability analysis failed: {e}")
            return {'error': str(e)}

    def _perform_temporal_analysis(self, X: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Perform temporal analysis."""
        try:
            _LOGGER.info("⏰ Performing temporal feature importance analysis...")
            
            temporal_result = self.temporal_analyzer.analyze_temporal_feature_importance(
                X, y, feature_names
            )
            
            _LOGGER.info("✅ Temporal analysis completed")
            return temporal_result
            
        except Exception as e:
            _LOGGER.error(f"❌ Temporal analysis failed: {e}")
            return {'error': str(e)}

    def _perform_causal_analysis(self, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str]) -> Dict[str, Any]:
        """Perform causal analysis."""
        try:
            _LOGGER.info("🔗 Performing causal pre-filtering...")
            
            causal_result = self.causal_analyzer.perform_causal_pre_filtering(
                X, y, feature_names
            )
            
            _LOGGER.info("✅ Causal analysis completed")
            return causal_result
            
        except Exception as e:
            _LOGGER.error(f"❌ Causal analysis failed: {e}")
            return {'error': str(e)}

    def _assess_selection_quality(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str],
                                selection_results: Dict[str, Any],
                                stability_results: Dict[str, Any],
                                temporal_results: Dict[str, Any],
                                causal_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of feature selection results."""
        try:
            # Get the best selection result (for now, use mRMR as default)
            best_selection = selection_results.get('mrmr', {})
            if not best_selection.get('success', False):
                # Find any successful selection
                for result in selection_results.values():
                    if result.get('success', False):
                        best_selection = result
                        break
            
            if not best_selection.get('success', False):
                _LOGGER.warning("⚠️ No successful selection results for quality assessment")
                return {'error': 'No successful selection results'}
            
            selected_features = best_selection.get('selected_features', [])
            
            # Calculate quality metrics
            quality_result = self.quality_calculator.calculate_comprehensive_quality_metrics(
                X, y, selected_features, feature_names, {
                    'bootstrap_stability': stability_results,
                    'temporal_stability': temporal_results,
                    'causal_analysis': causal_results
                }
            )
            
            _LOGGER.info(f"✅ Quality assessment completed - Overall score: {quality_result.get('overall_quality_score', 0):.3f}")
            return quality_result
            
        except Exception as e:
            _LOGGER.error(f"❌ Quality assessment failed: {e}")
            return {'error': str(e)}

    def _select_final_features(self, selection_results: Dict[str, Any],
                             stability_results: Dict[str, Any],
                             quality_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select final features based on all results."""
        try:
            # Collect all successful selections
            successful_selections = []
            for method_name, result in selection_results.items():
                if result.get('success', False):
                    selected_features = result.get('selected_features', [])
                    if selected_features:
                        successful_selections.append({
                            'method': method_name,
                            'features': selected_features,
                            'scores': result.get('scores', {}),
                            'stability_scores': stability_results.get(method_name, {}).get('stability_scores', {})
                        })
            
            if not successful_selections:
                _LOGGER.warning("⚠️ No successful selections for final feature selection")
                return {'selected_features': [], 'selected_indices': []}
            
            # Use consensus approach - features selected by multiple methods
            feature_votes = {}
            for selection in successful_selections:
                for feature in selection['features']:
                    if feature not in feature_votes:
                        feature_votes[feature] = {
                            'votes': 0,
                            'methods': [],
                            'scores': [],
                            'stability_scores': []
                        }
                    
                    feature_votes[feature]['votes'] += 1
                    feature_votes[feature]['methods'].append(selection['method'])
                    
                    # Add scores if available
                    if feature in selection.get('scores', {}):
                        feature_votes[feature]['scores'].append(selection['scores'][feature])
                    
                    # Add stability scores if available
                    if feature in selection.get('stability_scores', {}):
                        feature_votes[feature]['stability_scores'].append(selection['stability_scores'][feature])
            
            # Sort features by consensus (votes) and average scores
            def feature_score(feature):
                votes = feature_votes[feature]['votes']
                avg_score = np.mean(feature_votes[feature]['scores']) if feature_votes[feature]['scores'] else 0.0
                avg_stability = np.mean(feature_votes[feature]['stability_scores']) if feature_votes[feature]['stability_scores'] else 0.0
                
                # Weighted combination: votes (0.5) + scores (0.3) + stability (0.2)
                return votes * 0.5 + avg_score * 0.3 + avg_stability * 0.2
            
            sorted_features = sorted(feature_votes.keys(), key=feature_score, reverse=True)
            
            # Select top features (use a reasonable number)
            target_count = min(50, len(sorted_features))  # Cap at 50 features
            final_features = sorted_features[:target_count]
            
            # Get indices
            final_indices = []
            for feature in final_features:
                # Find feature index (this assumes feature_names is available in the calling context)
                # For now, we'll return empty indices
                pass
            
            result = {
                'selected_features': final_features,
                'selected_indices': final_indices,
                'feature_votes': feature_votes,
                'consensus_info': {
                    'n_methods': len(successful_selections),
                    'n_features_considered': len(feature_votes),
                    'n_features_selected': len(final_features),
                    'selection_methods': [s['method'] for s in successful_selections]
                }
            }
            
            _LOGGER.info(f"✅ Final feature selection completed - {len(final_features)} features selected")
            return result
            
        except Exception as e:
            _LOGGER.error(f"❌ Final feature selection failed: {e}")
            return {'selected_features': [], 'selected_indices': []}

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        try:
            summary = {
                'framework_version': '2.0.0',
                'components_initialized': {
                    'data_validator': hasattr(self, 'data_validator'),
                    'mrmr_selector': hasattr(self, 'mrmr_selector'),
                    'lasso_stability_selector': hasattr(self, 'lasso_stability_selector'),
                    'correlation_filter': hasattr(self, 'correlation_filter'),
                    'rfe_selector': hasattr(self, 'rfe_selector'),
                    'importance_ranker': hasattr(self, 'importance_ranker'),
                    'stability_analyzer': hasattr(self, 'stability_analyzer'),
                    'quality_calculator': hasattr(self, 'quality_calculator'),
                    'temporal_analyzer': hasattr(self, 'temporal_analyzer'),
                    'causal_analyzer': hasattr(self, 'causal_analyzer'),
                    'performance_monitor': hasattr(self, 'performance_monitor'),
                    'memory_optimizer': hasattr(self, 'memory_optimizer')
                },
                'configuration': self.config,
                'base_framework_stats': self.get_optimization_stats(),
                'system_requirements': self.check_system_requirements()
            }
            
            return summary
            
        except Exception as e:
            _LOGGER.error(f"❌ Pipeline summary generation failed: {e}")
            return {'error': str(e)}

    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive feature selection report."""
        try:
            report = f"""
=== Comprehensive Feature Selection Report ===
Generated: {datetime.now().isoformat()}
Framework Version: 2.0.0

=== Pipeline Summary ===
Execution Time: {results.get('pipeline_summary', {}).get('execution_time', 0):.3f}s
Original Data Shape: {results.get('pipeline_summary', {}).get('data_shape_original', 'Unknown')}
Cleaned Data Shape: {results.get('pipeline_summary', {}).get('data_shape_cleaned', 'Unknown')}
Features Removed: {results.get('pipeline_summary', {}).get('features_removed', 0)}
Target Features: {results.get('pipeline_summary', {}).get('target_features', 'Unknown')}
Final Feature Count: {results.get('pipeline_summary', {}).get('final_feature_count', 0)}
Model Type: {results.get('pipeline_summary', {}).get('model_type', 'Unknown')}

=== Final Selected Features ===
{', '.join(results.get('final_selected_features', []))}

=== Analysis Results ===
Stability Analysis: {'Enabled' if results.get('pipeline_summary', {}).get('stability_analysis_enabled', False) else 'Disabled'}
Temporal Analysis: {'Enabled' if results.get('pipeline_summary', {}).get('temporal_analysis_enabled', False) else 'Disabled'}
Causal Analysis: {'Enabled' if results.get('pipeline_summary', {}).get('causal_analysis_enabled', False) else 'Disabled'}

=== Quality Assessment ===
"""
            
            # Add quality results if available
            quality_results = results.get('quality_results', {})
            if quality_results and 'error' not in quality_results:
                overall_score = quality_results.get('overall_quality_score', 0)
                report += f"Overall Quality Score: {overall_score:.3f}\n"
                
                # Add individual metric scores
                for metric_name, metric_data in quality_results.items():
                    if isinstance(metric_data, dict) and 'error' not in metric_data:
                        if 'redundancy_score' in metric_data:
                            report += f"Redundancy Score: {metric_data['redundancy_score']:.3f}\n"
                        if 'relevance_score' in metric_data:
                            report += f"Relevance Score: {metric_data['relevance_score']:.3f}\n"
                        if 'stability_score' in metric_data:
                            report += f"Stability Score: {metric_data['stability_score']:.3f}\n"
                        if 'interpretability_score' in metric_data:
                            report += f"Interpretability Score: {metric_data['interpretability_score']:.3f}\n"
                        if 'performance_score' in metric_data:
                            report += f"Performance Score: {metric_data['performance_score']:.3f}\n"
            
            # Add selection method results
            report += "\n=== Selection Method Results ===\n"
            selection_results = results.get('selection_results', {})
            for method_name, method_result in selection_results.items():
                if method_result.get('success', False):
                    selected_count = len(method_result.get('selected_features', []))
                    execution_time = method_result.get('execution_time', 0)
                    report += f"{method_name}: {selected_count} features, {execution_time:.3f}s\n"
                else:
                    error = method_result.get('error', 'Unknown error')
                    report += f"{method_name}: Failed - {error}\n"
            
            return report
            
        except Exception as e:
            _LOGGER.error(f"❌ Comprehensive report generation failed: {e}")
            return f"Error generating comprehensive report: {e}"