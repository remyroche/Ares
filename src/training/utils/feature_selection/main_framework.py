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
    MRMRSelector, ElasticNetStabilitySelector,
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
    # Note: These are now consolidated into raw_data_exact_patterns below for better organization

    # Specific patterns for raw data columns - be more selective
    # Only exclude truly raw columns, keep derived features
    raw_data_specific_patterns = [
        '_time'  # Only exclude time columns, keep derived features
    ]

    # Raw data exact patterns that should be excluded - be more selective
    # Only exclude truly raw market data, keep derived features
    raw_data_exact_patterns = [
        'timestamp', 'open_time', 'close_time', 'first_trade_time', 'last_trade_time',
        'open', 'high', 'low', 'close', 'volume',  # Core OHLCV
        'symbol', 'exchange', 'market', 'pair',  # Metadata
        'target', 'label', 'y', 'model_score', 'prediction',  # Target/prediction columns
        'regime', 'regime_label', 'hmm_regime', 'cluster_regime'  # Regime labels
    ]

    # Define what we want to keep - derived features that contain these patterns
    keep_patterns = [
        'ratio', 'position', 'trend', 'strength', 'momentum', 'volatility',
        'return', 'log_return', 'range', 'size', 'pct', 'rolling', 'ma', 'sma', 'ema', 'wma'
    ]

    excluded_columns = []
    filtered_features = []

    for feature in feature_names:
        feature_lower = feature.lower()

        # Check for exact matches first (most restrictive)
        is_raw_data = feature_lower in raw_data_exact_patterns

        # If not an exact match, check for specific patterns at the end of column names
        # Be very selective - only exclude if it's clearly raw data
        if not is_raw_data:
            for pattern in raw_data_specific_patterns:
                if feature_lower.endswith(pattern):
                    # Only exclude time columns, keep derived features
                    if pattern == '_time':
                        # Only exclude if it's just 'timestamp' or similar raw time columns
                        if feature_lower in ['timestamp', 'open_time', 'close_time']:
                            is_raw_data = True
                            break

        # Special handling for regime columns - exclude any column containing regime
        if not is_raw_data and 'regime' in feature_lower:
            is_raw_data = True

        # Check if this feature contains keep patterns (derived features)
        is_derived_feature = False
        for keep_pattern in keep_patterns:
            if keep_pattern in feature_lower:
                is_derived_feature = True
                break

        # If it's not raw data but contains derived feature patterns, definitely keep it
        if not is_raw_data and is_derived_feature:
            filtered_features.append(feature)
        # If it's raw data, exclude it
        elif is_raw_data:
            excluded_columns.append(feature)
        # If it's not raw data and not obviously derived, be conservative and keep it
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

            # Pass mode information to elastic net stability selector for bootstrap count configuration
            elastic_net_config = self.config.get('elastic_net_stability', {})
            elastic_net_config['mode'] = self.config.get('mode', 'blank')
            self.elastic_net_stability_selector = ElasticNetStabilitySelector(elastic_net_config)

            # Correlation filter removed - functionality covered by mRMR and other methods
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
                    if result and 'selected_features' in result and result['selected_features']:
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
            
            # Step 3: mRMR Pre-filtering (remove 50% of excess features)
            _LOGGER.info("🔍 Step 3: mRMR pre-filtering (removing 50% of excess features)...")
            mrmr_prefilter_result = self._apply_mrmr_prefiltering(
                X_cleaned, y_cleaned, feature_names, target_features
            )
            
            # Extract filtered data and feature names
            X_filtered = mrmr_prefilter_result['filtered_data']
            y_filtered = mrmr_prefilter_result['filtered_target']
            filtered_feature_names = mrmr_prefilter_result['filtered_feature_names']
            mrmr_scores = mrmr_prefilter_result['mrmr_scores']
            
            _LOGGER.info(f"📊 mRMR pre-filtering: {len(filtered_feature_names)} features remaining from {len(feature_names)} original")
            
            # Step 4: Apply remaining selection methods on filtered data
            _LOGGER.info("🔍 Step 4: Applying remaining selection methods on filtered data...")
            selection_results = self._apply_remaining_selection_methods(
                X_filtered, y_filtered, filtered_feature_names, target_features
            )
            
            # Step 5: Stability analysis (if enabled)
            stability_results = {}
            if enable_stability_analysis:
                _LOGGER.info("📈 Step 5: Stability analysis...")
                stability_results = self._perform_stability_analysis(
                    X_filtered, y_filtered, filtered_feature_names, selection_results, target_features
                )
            
            # Step 6: Temporal analysis (if enabled)
            temporal_results = {}
            if enable_temporal_analysis:
                _LOGGER.info("⏰ Step 6: Temporal analysis...")
                temporal_results = self._perform_temporal_analysis(
                    X_filtered, y_filtered, filtered_feature_names
                )
            
            # Step 7: Causal analysis (if enabled)
            causal_results = {}
            if enable_causal_analysis:
                _LOGGER.info("🔗 Step 7: Causal analysis...")
                causal_results = self._perform_causal_analysis(
                    X_filtered, y_filtered, filtered_feature_names
                )
            
            # Step 8: PID analysis (if enabled)
            pid_results = {}
            if enable_pid_analysis:
                _LOGGER.info("🧮 Step 8: Partial Information Decomposition analysis...")
                pid_results = self._perform_pid_analysis(
                    X_filtered, y_filtered, filtered_feature_names
                )
            
            # Step 9: Calculate Statistical Score
            _LOGGER.info("📊 Step 9: Calculating Statistical Score...")
            statistical_score = self._calculate_statistical_score(
                stability_results, temporal_results, causal_results, pid_results, filtered_feature_names
            )
            
            # Step 10: Final feature selection with new scoring
            _LOGGER.info("🎯 Step 10: Final feature selection with new scoring...")
            final_selection = self._select_final_features_new_scoring(
                selection_results, statistical_score, mrmr_scores, filtered_feature_names
            )
            
            execution_time = time.time() - start_time
            
            # Compile comprehensive results
            result = {
                'selected_features': final_selection['selected_features'],
                'final_selected_features': final_selection['selected_features'],  # Keep for backward compatibility
                'final_selected_indices': final_selection['selected_indices'],
                'mrmr_prefilter_result': mrmr_prefilter_result,
                'selection_results': selection_results,
                'stability_results': stability_results,
                'temporal_results': temporal_results,
                'causal_results': causal_results,
                'pid_results': pid_results,
                'statistical_score': statistical_score,
                'final_scores': final_selection['final_scores'],
                'validation_result': validation_result,
                'cleaning_log': cleaning_log,
                'pipeline_summary': {
                    'execution_time': execution_time,
                    'target_features': target_features,
                    'final_feature_count': len(final_selection['selected_features']),
                    'data_shape_original': X.shape,
                    'data_shape_cleaned': X_cleaned.shape,
                    'data_shape_filtered': X_filtered.shape,
                    'features_removed_cleaning': cleaning_log.get('features_removed_count', 0),
                    'features_removed_mrmr': len(feature_names) - len(filtered_feature_names),
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
            
            # Generate CSV output
            csv_path = self._generate_csv_output(result, filtered_feature_names)
            if csv_path:
                _LOGGER.info(f"📄 CSV output generated: {csv_path}")
                result['csv_output_path'] = csv_path
            
            return result
            
        except Exception as e:
            _LOGGER.error(f"❌ Comprehensive feature selection failed: {e}")
            return {
                'selected_features': [],
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

    def _apply_mrmr_prefiltering(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str], target_features: int) -> Dict[str, Any]:
        """Apply mRMR pre-filtering to remove 50% of excess features (difference between initial and target)."""
        try:
            n_initial_features = len(feature_names)
            n_excess_features = max(0, n_initial_features - target_features)
            n_features_to_remove = n_excess_features // 2  # Remove 50% of excess features
            
            _LOGGER.info(f"🔍 Applying mRMR pre-filtering...")
            _LOGGER.info(f"📊 Initial features: {n_initial_features}, Target: {target_features}")
            _LOGGER.info(f"📊 Excess features: {n_excess_features}, Removing: {n_features_to_remove}")
            
            # If no excess features, return all features
            if n_features_to_remove == 0:
                _LOGGER.info("📊 No excess features to remove, keeping all features")
                return {
                    'filtered_data': X,
                    'filtered_target': y,
                    'filtered_feature_names': feature_names,
                    'mrmr_scores': {},
                    'removed_features': [],
                    'success': True
                }
            
            # Calculate mRMR scores for all features
            mrmr_result = self.mrmr_selector.select_features(X, y, feature_names, len(feature_names))
            
            if not mrmr_result.get('success', False):
                _LOGGER.warning("⚠️ mRMR pre-filtering failed, using all features")
                return {
                    'filtered_data': X,
                    'filtered_target': y,
                    'filtered_feature_names': feature_names,
                    'mrmr_scores': {},
                    'removed_features': [],
                    'success': False
                }
            
            # Get mRMR scores for all features
            mrmr_scores = mrmr_result.get('scores', {})
            
            # Sort features by mRMR score (ascending - worst first)
            sorted_features = sorted(mrmr_scores.items(), key=lambda x: x[1])
            
            # Remove bottom N worst features (50% of excess)
            removed_features = [feat for feat, score in sorted_features[:n_features_to_remove]]
            kept_features = [feat for feat, score in sorted_features[n_features_to_remove:]]
            
            # Create filtered dataset
            kept_indices = [feature_names.index(feat) for feat in kept_features]
            X_filtered = X[:, kept_indices]
            
            # Create filtered mRMR scores (only for kept features)
            filtered_mrmr_scores = {feat: mrmr_scores[feat] for feat in kept_features}
            
            _LOGGER.info(f"📊 mRMR pre-filtering: Removed {len(removed_features)} worst features, kept {len(kept_features)} best features")
            
            return {
                'filtered_data': X_filtered,
                'filtered_target': y,
                'filtered_feature_names': kept_features,
                'mrmr_scores': filtered_mrmr_scores,
                'removed_features': removed_features,
                'success': True
            }
            
        except Exception as e:
            _LOGGER.error(f"❌ mRMR pre-filtering failed: {e}")
            return {
                'filtered_data': X,
                'filtered_target': y,
                'filtered_feature_names': feature_names,
                'mrmr_scores': {},
                'removed_features': [],
                'success': False
            }

    def _apply_remaining_selection_methods(self, X: np.ndarray, y: np.ndarray, 
                                         feature_names: List[str], target_features: int) -> Dict[str, Any]:
        """Apply remaining selection methods (excluding mRMR and correlation filtering)."""
        try:
            selection_results = {}
            
            # Elastic Net stability selection
            _LOGGER.info("🔍 Applying Elastic Net stability selection...")
            elastic_net_result = self.elastic_net_stability_selector.select_features(X, y, feature_names)
            selection_results['elastic_net_stability'] = elastic_net_result
            
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
            
            _LOGGER.info(f"✅ Applied {len(selection_results)} remaining selection methods")
            return selection_results
            
        except Exception as e:
            _LOGGER.error(f"❌ Remaining selection methods application failed: {e}")
            return {'error': str(e)}

    def _perform_stability_analysis(self, X: np.ndarray, y: np.ndarray,
                                  feature_names: List[str],
                                  selection_results: Dict[str, Any], target_features: int) -> Dict[str, Any]:
        """Perform stability analysis."""
        try:
            stability_results = {}
            
            # Bootstrap stability analysis for each successful method
            for method_name, result in selection_results.items():
                if result.get('success', False):
                    _LOGGER.info(f"📈 Analyzing stability for {method_name}...")
                    
                    # Create selection method wrapper
                    def selection_wrapper(X_sub, y_sub, feature_names_sub, **kwargs):
                        if method_name == 'elastic_net_stability':
                            return self.elastic_net_stability_selector.select_features(X_sub, y_sub, feature_names_sub)
                        elif method_name == 'feature_importance':
                            return self.importance_ranker.select_features(X_sub, y_sub, feature_names_sub, kwargs.get('n_features', target_features))
                        elif method_name == 'rfe':
                            return self.rfe_selector.select_features(X_sub, y_sub, feature_names_sub, kwargs.get('n_features', target_features))
                        else:
                            return {'selected_features': [], 'success': False}
                    
                    # Perform bootstrap stability analysis
                    bootstrap_result = self.stability_analyzer.analyze_bootstrap_stability(
                        X, y, feature_names, selection_wrapper, {'n_features': target_features}
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
            
            # Calculate interaction scores for each feature
            interaction_scores = self.pid_decompositor.calculate_interaction_scores(pid_result, feature_names)
            
            # Generate artifacts with datetime
            artifacts = self.pid_decompositor.create_comprehensive_artifact(
                X, y, feature_names, pid_result, output_dir="pid_artifacts"
            )
            
            # Extract key information
            pid_analysis = {
                'redundancy_scores': pid_result.redundancy,
                'synergy_scores': pid_result.synergy,
                'unique_info_scores': pid_result.unique_info,
                'interaction_scores': interaction_scores,  # NEW: Individual feature scores
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
            _LOGGER.info(f"📊 Calculated interaction scores for {len(interaction_scores)} features")
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
                'interaction_scores': {},
                'polynomial_features': [],
                'interaction_features': [],
                'cross_timeframe_features': [],
                'significant_interactions': 0,
                'feature_pairs_analyzed': 0,
                'execution_time': 0.0,
                'error': str(e),
                'success': False
            }

    def _calculate_statistical_score(self, stability_results: Dict[str, Any],
                                   temporal_results: Dict[str, Any],
                                   causal_results: Dict[str, Any],
                                   pid_results: Dict[str, Any],
                                   feature_names: List[str]) -> Dict[str, float]:
        """Calculate statistical score from available analyses with equal weighting."""
        try:
            statistical_scores = {}
            available_analyses = []
            
            # Initialize all features with 0 score
            for feature in feature_names:
                statistical_scores[feature] = 0.0
            
            # Check which analyses are actually available and provide scores
            analysis_weights = {}
            
            # Stability analysis contribution (if available and provides scores)
            if stability_results and 'error' not in stability_results:
                for method_name, method_results in stability_results.items():
                    if method_results.get('success', False) and 'stability_scores' in method_results:
                        stability_scores = method_results.get('stability_scores', {})
                        if stability_scores:  # Only if we actually have scores
                            available_analyses.append('stability')
                            for feature, score in stability_scores.items():
                                if feature in statistical_scores:
                                    statistical_scores[feature] += score
                            break  # Use first successful method
            
            # Causal analysis contribution (if available and provides scores)
            if causal_results and 'error' not in causal_results and 'causal_scores' in causal_results:
                causal_scores = causal_results.get('causal_scores', {})
                if causal_scores:  # Only if we actually have scores
                    available_analyses.append('causal')
                    for feature, score in causal_scores.items():
                        if feature in statistical_scores:
                            statistical_scores[feature] += score
            
            # Temporal analysis contribution (if available and provides scores)
            if temporal_results and 'error' not in temporal_results and 'temporal_scores' in temporal_results:
                temporal_scores = temporal_results.get('temporal_scores', {})
                if temporal_scores:  # Only if we actually have scores
                    available_analyses.append('temporal')
                    for feature, score in temporal_scores.items():
                        if feature in statistical_scores:
                            statistical_scores[feature] += score
            
            # PID analysis contribution (if available and provides scores)
            if pid_results and 'error' not in pid_results and 'interaction_scores' in pid_results:
                interaction_scores = pid_results.get('interaction_scores', {})
                if interaction_scores:  # Only if we actually have scores
                    available_analyses.append('pid')
                    for feature, score in interaction_scores.items():
                        if feature in statistical_scores:
                            statistical_scores[feature] += score
            
            # Equal weighting for available analyses
            n_available = len(available_analyses)
            if n_available > 0:
                for feature in statistical_scores:
                    statistical_scores[feature] /= n_available  # Equal weighting
                
                # Normalize scores to 0-1 range
                max_score = max(statistical_scores.values())
                min_score = min(statistical_scores.values())
                if max_score > min_score:
                    for feature in statistical_scores:
                        statistical_scores[feature] = (statistical_scores[feature] - min_score) / (max_score - min_score)
            else:
                # If no analyses provide scores, use uniform distribution
                for feature in statistical_scores:
                    statistical_scores[feature] = 0.5  # Neutral score
            
            _LOGGER.info(f"📊 Statistical scores calculated for {len(statistical_scores)} features")
            _LOGGER.info(f"📊 Available analyses: {available_analyses} (equal weighting)")
            if len(available_analyses) == 0:
                _LOGGER.info(f"📊 Note: No analyses provided individual feature scores, using neutral scores")
            
            return statistical_scores
            
        except Exception as e:
            _LOGGER.error(f"❌ Statistical score calculation failed: {e}")
            return {feature: 0.5 for feature in feature_names}  # Neutral score on error

    def _select_final_features_new_scoring(self, selection_results: Dict[str, Any],
                                         statistical_score: Dict[str, float],
                                         mrmr_scores: Dict[str, float],
                                         feature_names: List[str]) -> Dict[str, Any]:
        """Select final features using new scoring system."""
        try:
            # Collect scores from each method
            elastic_net_scores = {}
            feature_importance_scores = {}
            rfe_scores = {}
            
            # Extract Elastic Net scores
            if selection_results.get('elastic_net_stability', {}).get('success', False):
                elastic_net_result = selection_results['elastic_net_stability']
                stability_scores = elastic_net_result.get('stability_scores', {})
                for feature, score in stability_scores.items():
                    elastic_net_scores[feature] = score
            
            # Extract Feature Importance scores
            if selection_results.get('feature_importance', {}).get('success', False):
                importance_result = selection_results['feature_importance']
                importance_scores = importance_result.get('importance_scores', {})
                for feature, score in importance_scores.items():
                    feature_importance_scores[feature] = score
            
            # Extract RFE scores (convert rankings to scores)
            if selection_results.get('rfe', {}).get('success', False):
                rfe_result = selection_results['rfe']
                rankings = rfe_result.get('feature_rankings', {})
                max_rank = max(rankings.values()) if rankings else 1
                for feature, rank in rankings.items():
                    # Convert ranking to score (lower rank = higher score)
                    rfe_scores[feature] = 1.0 - (rank - 1) / max(1, max_rank - 1)
            
            # Calculate final scores for all features
            final_scores = {}
            for feature in feature_names:
                # New scoring: Statistical (25%) + Elastic Net (25%) + Feature Importance (25%) + RFE (25%)
                stat_score = statistical_score.get(feature, 0.0)
                en_score = elastic_net_scores.get(feature, 0.0)
                fi_score = feature_importance_scores.get(feature, 0.0)
                rfe_score = rfe_scores.get(feature, 0.0)
                
                final_score = (stat_score * 0.25 + 
                             en_score * 0.25 + 
                             fi_score * 0.25 + 
                             rfe_score * 0.25)
                
                final_scores[feature] = final_score
            
            # Sort features by final score
            sorted_features = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features (cap at 50)
            target_count = min(50, len(sorted_features))
            selected_features = [feat for feat, score in sorted_features[:target_count]]
            selected_indices = [feature_names.index(feat) for feat in selected_features]
            
            # Create detailed results
            result = {
                'selected_features': selected_features,
                'selected_indices': selected_indices,
                'final_scores': final_scores,
                'method_scores': {
                    'statistical_score': statistical_score,
                    'elastic_net_scores': elastic_net_scores,
                    'feature_importance_scores': feature_importance_scores,
                    'rfe_scores': rfe_scores,
                    'mrmr_scores': mrmr_scores  # Include for reference but not in final scoring
                },
                'scoring_info': {
                    'statistical_weight': 0.25,
                    'elastic_net_weight': 0.25,
                    'feature_importance_weight': 0.25,
                    'rfe_weight': 0.25,
                    'mrmr_excluded_from_final': True
                }
            }
            
            _LOGGER.info(f"✅ Final feature selection completed - {len(selected_features)} features selected")
            _LOGGER.info(f"📊 Scoring weights: Statistical (25%), Elastic Net (25%), Feature Importance (25%), RFE (25%)")
            _LOGGER.info(f"📊 mRMR scores included for reference but excluded from final scoring")
            
            return result
            
        except Exception as e:
            _LOGGER.error(f"❌ Final feature selection failed: {e}")
            return {
                'selected_features': [],
                'selected_indices': [],
                'final_scores': {},
                'method_scores': {},
                'error': str(e)
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
                    'elastic_net_stability_selector': hasattr(self, 'elastic_net_stability_selector'),
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

    def _generate_csv_output(self, results: Dict[str, Any], feature_names: List[str]) -> Optional[str]:
        """Generate CSV output with all feature scores and final ranking."""
        try:
            import csv
            import os
            from pathlib import Path

            # Create outcomes directory if it doesn't exist
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp following outcomes naming convention
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"market_analysis_feature_selection_outcome_{timestamp}.csv"
            csv_path = outcomes_dir / csv_filename
            
            # Extract all scores
            final_scores = results.get('final_scores', {})
            method_scores = results.get('method_scores', {})
            
            mrmr_scores = method_scores.get('mrmr_scores', {})
            statistical_scores = method_scores.get('statistical_score', {})
            elastic_net_scores = method_scores.get('elastic_net_scores', {})
            feature_importance_scores = method_scores.get('feature_importance_scores', {})
            rfe_scores = method_scores.get('rfe_scores', {})
            
            # Extract individual analysis scores for detailed CSV
            stability_scores = {}
            temporal_scores = {}
            causal_scores = {}
            pid_scores = {}
            
            # Get individual scores from results
            if 'stability_results' in results:
                for method_name, method_results in results['stability_results'].items():
                    if method_results.get('success', False) and 'stability_scores' in method_results:
                        stability_scores = method_results.get('stability_scores', {})
                        break
            
            if 'temporal_results' in results and 'temporal_scores' in results['temporal_results']:
                temporal_scores = results['temporal_results'].get('temporal_scores', {})
            
            if 'causal_results' in results and 'causal_scores' in results['causal_results']:
                causal_scores = results['causal_results'].get('causal_scores', {})
            
            if 'pid_results' in results and 'interaction_scores' in results['pid_results']:
                pid_scores = results['pid_results'].get('interaction_scores', {})
            
            # Get final ranking
            final_ranking = {}
            if final_scores:
                sorted_features = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (feature, score) in enumerate(sorted_features, 1):
                    final_ranking[feature] = rank
            
            # Write CSV file
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'feature_name',
                    'final_score',
                    'final_ranking',
                    'mrmr_score',
                    'statistical_score',
                    'elastic_net_score',
                    'feature_importance_score',
                    'rfe_score',
                    'stability_score',
                    'temporal_score',
                    'causal_score',
                    'pid_interaction_score',
                    'selected'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write data for all features
                selected_features = set(results.get('final_selected_features', []))
                
                for feature in feature_names:
                    row = {
                        'feature_name': feature,
                        'final_score': final_scores.get(feature, 0.0),
                        'final_ranking': final_ranking.get(feature, 999),
                        'mrmr_score': mrmr_scores.get(feature, 0.0),
                        'statistical_score': statistical_scores.get(feature, 0.0),
                        'elastic_net_score': elastic_net_scores.get(feature, 0.0),
                        'feature_importance_score': feature_importance_scores.get(feature, 0.0),
                        'rfe_score': rfe_scores.get(feature, 0.0),
                        'stability_score': stability_scores.get(feature, 0.0),
                        'temporal_score': temporal_scores.get(feature, 0.0),
                        'causal_score': causal_scores.get(feature, 0.0),
                        'pid_interaction_score': pid_scores.get(feature, 0.0),
                        'selected': 'Yes' if feature in selected_features else 'No'
                    }
                    writer.writerow(row)
            
            # Also create a summary CSV with pipeline information
            summary_csv_path = outcomes_dir / f"feature_selection_summary_{timestamp}.csv"
            with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['metric', 'value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                pipeline_summary = results.get('pipeline_summary', {})
                summary_data = [
                    {'metric': 'execution_time_seconds', 'value': pipeline_summary.get('execution_time', 0)},
                    {'metric': 'original_features', 'value': pipeline_summary.get('data_shape_original', (0, 0))[1]},
                    {'metric': 'features_after_cleaning', 'value': pipeline_summary.get('data_shape_cleaned', (0, 0))[1]},
                    {'metric': 'features_after_mrmr_filtering', 'value': pipeline_summary.get('data_shape_filtered', (0, 0))[1]},
                    {'metric': 'features_removed_cleaning', 'value': pipeline_summary.get('features_removed_cleaning', 0)},
                    {'metric': 'features_removed_mrmr', 'value': pipeline_summary.get('features_removed_mrmr', 0)},
                    {'metric': 'final_selected_features', 'value': pipeline_summary.get('final_feature_count', 0)},
                    {'metric': 'model_type', 'value': pipeline_summary.get('model_type', 'unknown')},
                    {'metric': 'stability_analysis_enabled', 'value': pipeline_summary.get('stability_analysis_enabled', False)},
                    {'metric': 'temporal_analysis_enabled', 'value': pipeline_summary.get('temporal_analysis_enabled', False)},
                    {'metric': 'causal_analysis_enabled', 'value': pipeline_summary.get('causal_analysis_enabled', False)},
                    {'metric': 'pid_analysis_enabled', 'value': pipeline_summary.get('pid_analysis_enabled', False)},
                    {'metric': 'scoring_weights', 'value': 'Statistical(25%), ElasticNet(25%), FeatureImportance(25%), RFE(25%)'},
                    {'metric': 'mrmr_excluded_from_final', 'value': 'Yes (used for pre-filtering only)'},
                    {'metric': 'statistical_score_note', 'value': 'Based on available analyses (stability, temporal, causal, pid) with equal weighting'},
                    {'metric': 'temporal_analysis_implemented', 'value': 'Yes - provides individual feature scores'},
                    {'metric': 'pid_analysis_implemented', 'value': 'Yes - provides individual feature scores'}
                ]
                
                for row in summary_data:
                    writer.writerow(row)
            
            _LOGGER.info(f"📄 Generated CSV files:")
            _LOGGER.info(f"  - Detailed results: {csv_path}")
            _LOGGER.info(f"  - Summary: {summary_csv_path}")
            
            return csv_path
            
        except Exception as e:
            _LOGGER.error(f"❌ CSV generation failed: {e}")
            return None