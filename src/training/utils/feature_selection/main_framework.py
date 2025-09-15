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
from .partial_information_decompositor import PartialInformationDecompositor, PIDConfig


def filter_raw_market_data_columns(feature_names: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filter out raw market data columns that should not be considered as features for ML.

    Args:
        feature_names: List of all column/feature names

    Returns:
        Tuple of (filtered_feature_names, excluded_columns)
    """
    # Raw market data columns that should be excluded from feature selection
    # These are exact column names or very specific patterns for raw OHLCV data
    raw_data_patterns = [
        # Exact timestamp columns
        'timestamp', 'open_time', 'close_time', 'first_trade_time', 'last_trade_time',

        # Exact OHLC columns
        'open', 'high', 'low', 'close',

        # Exact volume columns (raw data)
        'volume', 'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume',
        'taker_sell_volume', 'taker_sell_quote_volume', 'total_volume',

        # Exact trade count columns
        'trades', 'taker_buy_trades', 'taker_sell_trades', 'total_trades',

        # Exact price columns (raw data)
        'price', 'avg_price', 'weighted_avg_price', 'last_price',

        # Return columns (these are often perfectly correlated with close)
        'close_return', 'close_log_return', 'open_return', 'high_return', 'low_return',

        # Basic market data identifiers
        'symbol', 'exchange', 'market', 'pair',

        # Target/label columns that shouldn't be features
        'target', 'label', 'y', 'model_score', 'prediction',

        # Regime-related columns (to avoid circular dependency)
        'regime', 'regime_label', 'hmm_regime', 'cluster_regime'
    ]

    # Specific patterns for raw data columns (more restrictive)
    raw_data_specific_patterns = [
        '_time', '_volume', '_trades', '_price', '_return', '_log_return'
    ]

    excluded_columns = []
    filtered_features = []

    for feature in feature_names:
        feature_lower = feature.lower()

        # Check for exact matches first (most restrictive)
        is_raw_data = feature_lower in raw_data_patterns

        # If not an exact match, check for specific patterns at the end of column names
        if not is_raw_data:
            for pattern in raw_data_specific_patterns:
                if feature_lower.endswith(pattern):
                    # Only exclude if it's a raw data pattern (not derived features)
                    # For example, exclude 'volume' but keep 'volume_ratio'
                    if pattern in ['_time', '_volume', '_trades', '_price', '_return', '_log_return']:
                        is_raw_data = True
                        break

        # Special handling for regime columns - exclude any column containing regime
        if not is_raw_data and 'regime' in feature_lower:
            is_raw_data = True

        if is_raw_data:
            excluded_columns.append(feature)
        else:
            filtered_features.append(feature)

    return filtered_features, excluded_columns

# Enhanced dependency management
try:
    from src.utils.logger import get_logger
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

            # Pass mode information to lasso stability selector for bootstrap count configuration
            lasso_config = self.config.get('lasso_stability', {})
            lasso_config['mode'] = self.config.get('mode', 'blank')
            self.lasso_stability_selector = LassoStabilitySelector(lasso_config)

            self.correlation_filter = CorrelationBasedFilter(self.config.get('correlation_filter', {}))
            self.rfe_selector = RecursiveFeatureEliminator(self.config.get('rfe', {}))
            self.importance_ranker = FeatureImportanceRanker(self.config.get('importance', {}))
            _LOGGER.info("✅ Selection methods initialized")
            
            # Analysis components
            # Pass mode information to stability analyzer for bootstrap count configuration
            stability_config = self.config.get('stability_analysis', {})
            stability_config['mode'] = self.config.get('mode', 'blank')
            self.stability_analyzer = StabilityAnalyzer(stability_config)
            self.quality_calculator = QualityMetricsCalculator(self.config.get('quality_metrics', {}))
            self.temporal_analyzer = TemporalAnalyzer(self.config.get('temporal_analysis', {}))
            self.causal_analyzer = CausalAnalyzer(self.config.get('causal_analysis', {}))
            
            # PID decompositor component
            pid_config = PIDConfig(**self.config.get('partial_information_decompositor', {}))
            self.pid_decompositor = PartialInformationDecompositor(pid_config)
            _LOGGER.info("✅ Analysis components initialized")
            
            # Performance monitoring
            self.performance_monitor = PerformanceMonitor(max_history=1000)
            self.memory_optimizer = MemoryOptimizer(self.config.get('memory_optimization', {}))
            _LOGGER.info("✅ Performance monitoring initialized")
            
        except Exception as e:
            _LOGGER.error(f"❌ Component initialization failed: {e}")
            raise

    def select_features(self, X: pd.DataFrame, y: np.ndarray,
                       method: str = 'comprehensive',
                       max_features: Optional[int] = None,
                       is_classification: bool = True) -> Dict[str, Any]:
        """
        Main feature selection interface compatible with HMM training pipeline.

        Args:
            X: Feature matrix (DataFrame)
            y: Target vector
            method: Selection method ('comprehensive', 'fast', 'basic')
            max_features: Maximum number of features to select
            is_classification: Whether this is a classification task

        Returns:
            Dictionary with 'selected_features' key containing selected feature names
        """
        try:
            _LOGGER.info(f"🔍 Starting feature selection with method: {method}")

            # Convert DataFrame to numpy array and get feature names
            if isinstance(X, pd.DataFrame):
                original_feature_names = X.columns.tolist()
                X_array = X.values
            else:
                original_feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                X_array = X

            # Step 1: Filter out raw market data columns that shouldn't be features
            _LOGGER.info("🔍 Filtering out raw market data columns...")
            filtered_features, excluded_columns = filter_raw_market_data_columns(original_feature_names)

            if excluded_columns:
                _LOGGER.info(f"📊 Excluded {len(excluded_columns)} raw market data columns: {excluded_columns[:10]}{'...' if len(excluded_columns) > 10 else ''}")
                _LOGGER.info(f"📊 Keeping {len(filtered_features)} potential features for selection")

                # Create filtered dataset
                if isinstance(X, pd.DataFrame):
                    X_filtered = X[filtered_features]
                    feature_names = filtered_features
                else:
                    # For numpy arrays, we need to filter columns
                    feature_indices = [original_feature_names.index(feat) for feat in filtered_features]
                    X_filtered = X_array[:, feature_indices]
                    feature_names = filtered_features
                    X_array = X_filtered
            else:
                _LOGGER.info("📊 No raw data columns found to exclude")
                feature_names = original_feature_names
                X_filtered = X
                if not isinstance(X, pd.DataFrame):
                    X_array = X_filtered

            # Use comprehensive feature selection if available
            try:
                if hasattr(self, 'run_comprehensive_feature_selection'):
                    result = self.run_comprehensive_feature_selection(
                        X_array, y,
                        feature_names,
                        target_features=max_features
                    )
                    if result and 'selected_features' in result:
                        selected_features = result['selected_features']
                        _LOGGER.info(f"✅ Feature selection completed (comprehensive): {len(selected_features)} features selected")
                    else:
                        # Fallback to simple selection
                        if max_features and max_features < len(feature_names):
                            selected_features = feature_names[:max_features]
                        else:
                            selected_features = feature_names
                        _LOGGER.info(f"✅ Feature selection completed (simple fallback): {len(selected_features)} features selected")
                else:
                    # Simple fallback
                    if max_features and max_features < len(feature_names):
                        selected_features = feature_names[:max_features]
                    else:
                        selected_features = feature_names
                    _LOGGER.info(f"✅ Feature selection completed (simple): {len(selected_features)} features selected")
            except Exception as cache_error:
                _LOGGER.warning(f"⚠️ Comprehensive selection failed ({cache_error}), using simple fallback")
                # Fallback to simple selection
                if max_features and max_features < len(feature_names):
                    selected_features = feature_names[:max_features]
                else:
                    selected_features = feature_names
                _LOGGER.info(f"✅ Feature selection completed (fallback): {len(selected_features)} features selected")

            return {
                'selected_features': selected_features,
                'method': method,
                'total_features': len(feature_names),
                'selected_count': len(selected_features),
                'selection_details': {'fallback': True, 'reason': 'Cache implementation issue'},
                'fallback': True
            }

        except Exception as e:
            _LOGGER.error(f"❌ Feature selection failed: {e}")
            # Fallback: return all features
            all_features = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
            return {
                'selected_features': all_features[:max_features] if max_features else all_features,
                'method': method,
                'error': str(e),
                'fallback': True
            }

    def run_comprehensive_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                          feature_names: List[str],
                                          target_features: Optional[int] = None,
                                          model_type: str = 'default',
                                          enable_stability_analysis: bool = True,
                                          enable_temporal_analysis: bool = False,
                                          enable_causal_analysis: bool = False,
                                          enable_pid_analysis: bool = False) -> Dict[str, Any]:
        """Run comprehensive feature selection pipeline."""
        start_time = time.time()
        _LOGGER.info(f"🚀 Starting comprehensive feature selection pipeline...")
        _LOGGER.info(f"📊 Parameters - Data shape: {X.shape}, Target features: {target_features}")

        try:
            # Step 1: Data validation and cleaning (on pre-filtered data)
            _LOGGER.info("🔍 Step 1: Data validation and cleaning...")
            _LOGGER.info("📊 Note: Raw market data columns have been pre-filtered before this step")

            validation_result = self.data_validator.validate_data_quality(X, y, feature_names)

            if not validation_result.get('is_valid', True):
                _LOGGER.warning(f"⚠️ Data validation issues: {validation_result.get('issues', [])}")
            else:
                _LOGGER.info("✅ Data validation passed - no major issues detected")

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
            
            # Step 7: PID analysis (if enabled)
            pid_results = {}
            if enable_pid_analysis:
                _LOGGER.info("🧮 Step 7: Partial Information Decomposition analysis...")
                pid_results = self._perform_pid_analysis(
                    X_cleaned, y_cleaned, feature_names
                )
            
            # Step 8: Quality assessment
            _LOGGER.info("📊 Step 8: Quality assessment...")
            quality_results = self._assess_selection_quality(
                X_cleaned, y_cleaned, feature_names, selection_results,
                stability_results, temporal_results, causal_results, pid_results
            )
            
            # Step 9: Final feature selection
            _LOGGER.info("🎯 Step 9: Final feature selection...")
            final_selection = self._select_final_features(
                selection_results, stability_results, quality_results, pid_results
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
                'pid_results': pid_results,
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
                    'causal_analysis_enabled': enable_causal_analysis,
                    'pid_analysis_enabled': enable_pid_analysis
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
                'pid_results': {},
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
                                causal_results: Dict[str, Any],
                                pid_results: Dict[str, Any] = None) -> Dict[str, Any]:
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

    def _perform_pid_analysis(self, X: np.ndarray, y: np.ndarray, 
                            feature_names: List[str]) -> Dict[str, Any]:
        """Perform partial information decomposition analysis."""
        try:
            _LOGGER.info("🧮 Starting PID analysis...")
            
            # Run PID decomposition
            pid_result = self.pid_decompositor.decompose_information(X, y, feature_names)
            
            # Generate artifacts with datetime
            artifacts = self.pid_decompositor.create_comprehensive_artifact(
                X, y, feature_names, pid_result, output_dir="pid_artifacts"
            )
            
            # Extract key information
            pid_analysis = {
                'redundancy_scores': pid_result.redundancy,
                'synergy_scores': pid_result.synergy,
                'unique_info_scores': pid_result.unique_info,
                'polynomial_features': pid_result.polynomial_features,
                'interaction_features': pid_result.interaction_features,
                'cross_timeframe_features': pid_result.cross_timeframe_features,
                'significant_interactions': pid_result.significant_interactions,
                'feature_pairs_analyzed': pid_result.feature_pairs_analyzed,
                'execution_time': pid_result.execution_time,
                'artifacts_generated': artifacts,
                'success': True
            }
            
            _LOGGER.info(f"✅ PID analysis completed in {pid_result.execution_time:.3f}s")
            _LOGGER.info(f"📊 Found {pid_result.significant_interactions} significant interactions")
            _LOGGER.info(f"🔧 Generated {len(pid_result.polynomial_features)} polynomial features")
            _LOGGER.info(f"🔧 Generated {len(pid_result.interaction_features)} interaction features")
            _LOGGER.info(f"🔧 Generated {len(pid_result.cross_timeframe_features)} cross-timeframe features")
            
            return pid_analysis
            
        except Exception as e:
            _LOGGER.error(f"❌ PID analysis failed: {e}")
            return {
                'redundancy_scores': {},
                'synergy_scores': {},
                'unique_info_scores': {},
                'polynomial_features': [],
                'interaction_features': [],
                'cross_timeframe_features': [],
                'significant_interactions': 0,
                'feature_pairs_analyzed': 0,
                'execution_time': 0.0,
                'error': str(e),
                'success': False
            }

    def _select_final_features(self, selection_results: Dict[str, Any],
                             stability_results: Dict[str, Any],
                             quality_results: Dict[str, Any],
                             pid_results: Dict[str, Any] = None) -> Dict[str, Any]:
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