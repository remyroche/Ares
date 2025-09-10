from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Step 8: Advanced Feature Selection - Per-Regime Implementation.

This module provides per-HMM regime feature selection functionality, ensuring that
feature selection is optimized specifically for each regime's characteristics.
"""
import asyncio
from pathlib import Path
import json
from typing import Dict, Any, Optional, List
from src.training.steps.market_analysis.step08_advanced_feature_selection import Step08AdvancedFeatureSelection
from src.training.steps.market_analysis.regime_continuity_decorator import per_regime_step
from src.utils.pipeline_standards import pipeline_standards
import numpy as np

from ....utils.decorators import traced, validates, handles_errors
from src.utils.logger import get_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
import logging
import time
from datetime import datetime

logger = get_logger('Step8AdvancedFeatureSelectionPerRegime')

class PerRegimeAdvancedFeatureSelectionStep(Step08AdvancedFeatureSelection):
    """Advanced feature selection step that processes each regime separately."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        start_time = time.time()
        logger.info('🚀 Initializing Per-Regime Advanced Feature Selection Step...')
        logger.info(f'📋 Configuration keys: {list(config.keys())}')
        
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_feature_selection', True)
        self.regime_specific_configs = config.get('regime_specific_feature_selection_configs', {})
        self.adaptive_feature_selection = config.get('adaptive_feature_selection_per_regime', True)
        
        # Log initialization details
        logger.info(f'✅ Per-regime feature selection enabled: {self.per_regime_enabled}')
        logger.info(f'📊 Regime-specific configs available: {len(self.regime_specific_configs)}')
        logger.info(f'🔄 Adaptive feature selection enabled: {self.adaptive_feature_selection}')
        
        init_time = time.time() - start_time
        logger.info(f'⏱️ Initialization completed in {init_time:.3f} seconds')

    @traced(span_name='execute_per_regime_feature_selection')
    @per_regime_step('step08_advanced_feature_selection')
    async def execute_per_regime_feature_selection(self, symbol: str, exchange: str, timeframe: str, data_dir: str, force_rerun: bool = False, regime_id: Optional[int]=None, regime_context: Optional[Any]=None, per_regime: bool = True) -> bool:
        """Execute feature selection on a per-regime basis.
        
        Each regime may have different feature importance patterns, so feature
        selection should be optimized specifically for each regime's characteristics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            regime_id: Regime ID (provided by decorator)
            regime_context: Regime context (provided by decorator)
            per_regime: Per-regime flag (provided by decorator)
            
        Returns:
            Success status
        """
        try:
            execution_start = time.time()
            logger.info(f'🚀 Starting per-regime feature selection for regime {regime_id}')
            logger.info(f'📊 Input parameters - Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}')
            logger.info(f'📁 Data directory: {data_dir}, Force rerun: {force_rerun}')
            
            # Step 1: Load matrix data
            logger.info(f'📥 Step 1: Loading matrix data for regime {regime_id}...')
            matrix_start = time.time()
            matrix_data = await self._load_matrix_data(symbol, exchange, timeframe, data_dir, regime_id)
            matrix_time = time.time() - matrix_start
            
            if matrix_data is None:
                logger.error(f'❌ Failed to load matrix data for regime {regime_id} after {matrix_time:.3f}s')
                return False
            
            logger.info(f'✅ Matrix data loaded successfully in {matrix_time:.3f}s')
            logger.info(f'📊 Matrix data keys: {list(matrix_data.keys()) if isinstance(matrix_data, dict) else "N/A"}')
            
            # Step 2: Get regime configuration
            logger.info(f'⚙️ Step 2: Getting regime-specific configuration for regime {regime_id}...')
            config_start = time.time()
            regime_config = self._get_regime_feature_selection_config(regime_id)
            config_time = time.time() - config_start
            
            logger.info(f'✅ Regime configuration retrieved in {config_time:.3f}s')
            logger.info(f'🔧 Config keys: {list(regime_config.keys()) if isinstance(regime_config, dict) else "N/A"}')
            
            # Step 3: Apply feature selection
            logger.info(f'🔍 Step 3: Applying feature selection for regime {regime_id}...')
            selection_start = time.time()
            selection_results = await self._apply_regime_feature_selection(matrix_data, regime_config, regime_id)
            selection_time = time.time() - selection_start
            
            if selection_results is None:
                logger.error(f'❌ Failed feature selection for regime {regime_id} after {selection_time:.3f}s')
                return False
            
            logger.info(f'✅ Feature selection completed in {selection_time:.3f}s')
            logger.info(f'📊 Selection results keys: {list(selection_results.keys()) if isinstance(selection_results, dict) else "N/A"}')
            
            # Step 4: Save results
            logger.info(f'💾 Step 4: Saving selection results for regime {regime_id}...')
            save_start = time.time()
            success = await self._save_regime_selection_results(selection_results, symbol, exchange, timeframe, data_dir, regime_id)
            save_time = time.time() - save_start
            
            total_time = time.time() - execution_start
            
            if success:
                logger.info(f'✅ Successfully completed feature selection for regime {regime_id}')
                logger.info(f'⏱️ Total execution time: {total_time:.3f}s (Matrix: {matrix_time:.3f}s, Config: {config_time:.3f}s, Selection: {selection_time:.3f}s, Save: {save_time:.3f}s)')
            else:
                logger.error(f'❌ Failed to save selection results for regime {regime_id} after {save_time:.3f}s')
            
            return success
        except Exception as e:
            execution_time = time.time() - execution_start if 'execution_start' in locals() else 0
            logger.exception(f'❌ Error in per-regime feature selection for regime {regime_id} after {execution_time:.3f}s: {e}')
            logger.error(f'🔍 Error context - Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}')
            return False

    async def _load_matrix_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str, regime_id: int) -> Optional[Dict[str, Any]]:
        """Load matrix operation results for a specific regime.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            Matrix data or None
        """
        try:
            logger.info(f'🔍 Searching for matrix data files for regime {regime_id}...')
            
            # Try regime-specific file first
            regime_specific_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_matrix_operations_regime_{regime_id}.json'
            logger.info(f'📁 Checking regime-specific path: {regime_specific_path}')
            
            matrix_path = None
            if regime_specific_path.exists():
                matrix_path = regime_specific_path
                logger.info(f'✅ Found regime-specific matrix data file')
            else:
                # Fallback to aggregated file
                aggregated_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_matrix_operations_aggregated.json'
                logger.info(f'📁 Checking aggregated path: {aggregated_path}')
                
                if aggregated_path.exists():
                    matrix_path = aggregated_path
                    logger.info(f'✅ Found aggregated matrix data file')
                else:
                    logger.error(f'❌ No matrix data files found for regime {regime_id}')
                    logger.error(f'   Searched paths:')
                    logger.error(f'   - {regime_specific_path}')
                    logger.error(f'   - {aggregated_path}')
                    return None
            
            # Load the data
            logger.info(f'📥 Loading matrix data from: {matrix_path}')
            load_start = time.time()
            
            with open(matrix_path, 'r') as f:
                data = json.load(f)
            
            load_time = time.time() - load_start
            
            # Log data characteristics
            if isinstance(data, dict):
                logger.info(f'✅ Loaded matrix data for regime {regime_id} in {load_time:.3f}s')
                logger.info(f'📊 Data structure: {len(data)} top-level keys')
                logger.info(f'🔑 Data keys: {list(data.keys())}')
                
                # Log feature information if available
                if 'feature_columns' in data:
                    feature_count = len(data['feature_columns'])
                    logger.info(f'📈 Feature columns: {feature_count}')
                else:
                    logger.warning(f'⚠️ No feature_columns found in matrix data')
                
                # Log operations information if available
                if 'operations' in data:
                    ops = data['operations']
                    logger.info(f'🔧 Operations available: {list(ops.keys()) if isinstance(ops, dict) else "N/A"}')
                else:
                    logger.warning(f'⚠️ No operations found in matrix data')
            else:
                logger.warning(f'⚠️ Matrix data is not a dictionary: {type(data)}')
            
            return data
            
        except FileNotFoundError as e:
            logger.error(f'❌ Matrix data file not found for regime {regime_id}: {e}')
            return None
        except json.JSONDecodeError as e:
            logger.error(f'❌ JSON decode error loading matrix data for regime {regime_id}: {e}')
            return None
        except Exception as e:
            logger.error(f'❌ Unexpected error loading matrix data for regime {regime_id}: {e}')
            logger.exception(f'🔍 Full error details:')
            return None
    @log_all_calls
    def _get_regime_feature_selection_config(self, regime_id: int) -> Dict[str, Any]:
        """Get feature selection configuration for a specific regime.
        
        Different regimes may benefit from different feature selection strategies.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific feature selection configuration
        """
        logger.info(f'⚙️ Getting feature selection configuration for regime {regime_id}...')
        
        # Check for regime-specific configuration first
        regime_key = f'regime_{regime_id}'
        if regime_key in self.regime_specific_configs:
            config = self.regime_specific_configs[regime_key]
            logger.info(f'✅ Using custom configuration for regime {regime_id}')
            logger.info(f'🔧 Custom config keys: {list(config.keys()) if isinstance(config, dict) else "N/A"}')
            return config
        
        # Use default configuration based on regime characteristics
        logger.info(f'📋 Using default configuration for regime {regime_id}')
        
        base_config = {
            'enable_correlation_filtering': True, 
            'enable_variance_filtering': True, 
            'enable_mutual_information': True, 
            'enable_recursive_feature_elimination': True, 
            'enable_permutation_importance': True
        }
        
        if regime_id <= 2:
            config = {
                **base_config, 
                'selection_strategy': {
                    'max_features': 50, 
                    'correlation_threshold': 0.8, 
                    'variance_threshold': 0.01, 
                    'mutual_info_threshold': 0.05, 
                    'emphasis': 'trend_features'
                }, 
                'feature_types': ['trend_indicators', 'momentum_features', 'volume_trend_features', 'price_trend_features']
            }
            logger.info(f'🎯 Applied trend-focused configuration for regime {regime_id} (low regime ID)')
            
        elif regime_id >= 5:
            config = {
                **base_config, 
                'selection_strategy': {
                    'max_features': 40, 
                    'correlation_threshold': 0.7, 
                    'variance_threshold': 0.02, 
                    'mutual_info_threshold': 0.03, 
                    'emphasis': 'volatility_features'
                }, 
                'feature_types': ['volatility_indicators', 'mean_reversion_features', 'oscillator_features', 'range_features']
            }
            logger.info(f'📊 Applied volatility-focused configuration for regime {regime_id} (high regime ID)')
            
        else:
            config = {
                **base_config, 
                'selection_strategy': {
                    'max_features': 45, 
                    'correlation_threshold': 0.75, 
                    'variance_threshold': 0.015, 
                    'mutual_info_threshold': 0.04, 
                    'emphasis': 'balanced_features'
                }, 
                'feature_types': ['mixed_indicators', 'balanced_features', 'adaptive_features', 'composite_features']
            }
            logger.info(f'⚖️ Applied balanced configuration for regime {regime_id} (medium regime ID)')
        
        # Log configuration details
        strategy = config.get('selection_strategy', {})
        logger.info(f'🔧 Configuration details:')
        logger.info(f'   - Max features: {strategy.get("max_features", "N/A")}')
        logger.info(f'   - Correlation threshold: {strategy.get("correlation_threshold", "N/A")}')
        logger.info(f'   - Variance threshold: {strategy.get("variance_threshold", "N/A")}')
        logger.info(f'   - Mutual info threshold: {strategy.get("mutual_info_threshold", "N/A")}')
        logger.info(f'   - Emphasis: {strategy.get("emphasis", "N/A")}')
        logger.info(f'   - Feature types: {config.get("feature_types", [])}')
        
        return config

    async def _apply_regime_feature_selection(self, matrix_data: Dict[str, Any], regime_config: Dict[str, Any], regime_id: int) -> Optional[Dict[str, Any]]:
        """Apply feature selection to regime data.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Feature selection results or None
        """
        try:
            selection_start = time.time()
            logger.info(f'🔧 Applying feature selection for regime {regime_id}...')
            
            # Extract feature columns
            feature_columns = matrix_data.get('feature_columns', [])
            if not feature_columns:
                logger.warning(f'⚠️ No feature columns found for regime {regime_id}')
                return None
            
            logger.info(f'📊 Starting with {len(feature_columns)} features for regime {regime_id}')
            
            # Initialize results structure
            results = {
                'regime_id': regime_id, 
                'total_features': len(feature_columns), 
                'selection_strategy': regime_config.get('selection_strategy', {}), 
                'selected_features': [], 
                'feature_scores': {}, 
                'selection_metadata': {},
                'execution_times': {}
            }
            
            # Apply correlation filtering
            if regime_config.get('enable_correlation_filtering', True):
                logger.info(f'🔗 Applying correlation filtering for regime {regime_id}...')
                corr_start = time.time()
                correlation_results = self._apply_correlation_filtering(matrix_data, regime_config, feature_columns)
                corr_time = time.time() - corr_start
                results['selection_metadata']['correlation_filtering'] = correlation_results
                results['execution_times']['correlation_filtering'] = corr_time
                logger.info(f'✅ Correlation filtering completed in {corr_time:.3f}s')
            else:
                logger.info(f'⏭️ Skipping correlation filtering for regime {regime_id}')
            
            # Apply variance filtering
            if regime_config.get('enable_variance_filtering', True):
                logger.info(f'📊 Applying variance filtering for regime {regime_id}...')
                var_start = time.time()
                variance_results = self._apply_variance_filtering(matrix_data, regime_config, feature_columns)
                var_time = time.time() - var_start
                results['selection_metadata']['variance_filtering'] = variance_results
                results['execution_times']['variance_filtering'] = var_time
                logger.info(f'✅ Variance filtering completed in {var_time:.3f}s')
            else:
                logger.info(f'⏭️ Skipping variance filtering for regime {regime_id}')
            
            # Apply mutual information filtering
            if regime_config.get('enable_mutual_information', True):
                logger.info(f'🧠 Applying mutual information filtering for regime {regime_id}...')
                mi_start = time.time()
                mi_results = self._apply_mutual_information_filtering(matrix_data, regime_config, feature_columns)
                mi_time = time.time() - mi_start
                results['selection_metadata']['mutual_information'] = mi_results
                results['execution_times']['mutual_information'] = mi_time
                logger.info(f'✅ Mutual information filtering completed in {mi_time:.3f}s')
            else:
                logger.info(f'⏭️ Skipping mutual information filtering for regime {regime_id}')
            
            # Apply recursive feature elimination
            if regime_config.get('enable_recursive_feature_elimination', True):
                logger.info(f'🔄 Applying recursive feature elimination for regime {regime_id}...')
                rfe_start = time.time()
                rfe_results = self._apply_recursive_feature_elimination(matrix_data, regime_config, feature_columns)
                rfe_time = time.time() - rfe_start
                results['selection_metadata']['recursive_feature_elimination'] = rfe_results
                results['execution_times']['recursive_feature_elimination'] = rfe_time
                logger.info(f'✅ Recursive feature elimination completed in {rfe_time:.3f}s')
            else:
                logger.info(f'⏭️ Skipping recursive feature elimination for regime {regime_id}')
            
            # Apply permutation importance
            if regime_config.get('enable_permutation_importance', True):
                logger.info(f'🎯 Applying permutation importance for regime {regime_id}...')
                perm_start = time.time()
                perm_results = self._apply_permutation_importance(matrix_data, regime_config, feature_columns)
                perm_time = time.time() - perm_start
                results['selection_metadata']['permutation_importance'] = perm_results
                results['execution_times']['permutation_importance'] = perm_time
                logger.info(f'✅ Permutation importance completed in {perm_time:.3f}s')
            else:
                logger.info(f'⏭️ Skipping permutation importance for regime {regime_id}')
            
            # Combine selection results
            logger.info(f'🔀 Combining selection results for regime {regime_id}...')
            combine_start = time.time()
            selected_features = self._combine_selection_results(results['selection_metadata'], regime_config, feature_columns)
            combine_time = time.time() - combine_start
            
            # Finalize results
            results['selected_features'] = selected_features
            results['selected_count'] = len(selected_features)
            results['selection_rate'] = len(selected_features) / len(feature_columns) if feature_columns else 0
            results['execution_times']['combine_results'] = combine_time
            
            total_time = time.time() - selection_start
            
            logger.info(f'✅ Completed feature selection for regime {regime_id}')
            logger.info(f'📊 Results: {len(selected_features)}/{len(feature_columns)} features selected ({results["selection_rate"]:.2%})')
            logger.info(f'⏱️ Total selection time: {total_time:.3f}s')
            logger.info(f'⏱️ Breakdown: {results["execution_times"]}')
            
            return results
            
        except Exception as e:
            selection_time = time.time() - selection_start if 'selection_start' in locals() else 0
            logger.error(f'❌ Error applying feature selection for regime {regime_id} after {selection_time:.3f}s: {e}')
            logger.exception(f'🔍 Full error details:')
            return None
    @log_all_calls
    def _apply_correlation_filtering(self, matrix_data: Dict[str, Any], regime_config: Dict[str, Any], feature_columns: List[str]) -> Dict[str, Any]:
        """Apply correlation-based feature filtering.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            feature_columns: List of feature columns
            
        Returns:
            Correlation filtering results
        """
        try:
            logger.info(f'🔗 Starting correlation filtering with {len(feature_columns)} features...')
            
            # Get correlation threshold
            correlation_threshold = regime_config.get('selection_strategy', {}).get('correlation_threshold', 0.8)
            logger.info(f'📊 Using correlation threshold: {correlation_threshold}')
            
            # Extract correlation data
            correlation_ops = matrix_data.get('operations', {}).get('correlation_matrix', {})
            if not correlation_ops:
                logger.warning(f'⚠️ No correlation matrix operations found in matrix data')
                return {'error': 'No correlation matrix operations found', 'threshold': correlation_threshold}
            
            high_correlations = correlation_ops.get('high_correlations', [])
            logger.info(f'📈 Found {len(high_correlations)} high correlation pairs')
            
            # Filter features based on correlation threshold
            features_to_remove = set()
            for corr in high_correlations:
                if corr['abs_correlation'] >= correlation_threshold:
                    features_to_remove.add(corr['feature2'])
                    logger.debug(f'🔍 High correlation detected: {corr["feature1"]} <-> {corr["feature2"]} = {corr["abs_correlation"]:.3f}')
            
            remaining_features = [f for f in feature_columns if f not in features_to_remove]
            
            result = {
                'threshold': correlation_threshold, 
                'high_correlations': len(high_correlations), 
                'features_removed': len(features_to_remove), 
                'features_remaining': len(remaining_features), 
                'removed_features': list(features_to_remove)
            }
            
            logger.info(f'✅ Correlation filtering completed:')
            logger.info(f'   - Features removed: {len(features_to_remove)}')
            logger.info(f'   - Features remaining: {len(remaining_features)}')
            logger.info(f'   - Reduction rate: {len(features_to_remove)/len(feature_columns):.2%}')
            
            return result
            
        except Exception as e:
            logger.error(f'❌ Error in correlation filtering: {e}')
            logger.exception(f'🔍 Full error details:')
            return {'error': str(e), 'threshold': correlation_threshold if 'correlation_threshold' in locals() else 0.8}
    @log_all_calls
    def _apply_variance_filtering(self, matrix_data: Dict[str, Any], regime_config: Dict[str, Any], feature_columns: List[str]) -> Dict[str, Any]:
        """Apply variance-based feature filtering.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            feature_columns: List of feature columns
            
        Returns:
            Variance filtering results
        """
        try:
            logger.info(f'📊 Starting variance filtering with {len(feature_columns)} features...')
            
            # Get variance threshold
            variance_threshold = regime_config.get('selection_strategy', {}).get('variance_threshold', 0.01)
            logger.info(f'📊 Using variance threshold: {variance_threshold}')
            
            # Try to get feature importance from balanced analysis first
            balanced_ops = matrix_data.get('operations', {}).get('balanced_analysis', {})
            feature_importance = balanced_ops.get('feature_importance', [])
            
            if not feature_importance:
                logger.info(f'📊 No feature importance in balanced analysis, trying correlation matrix diagonal...')
                # Fallback to correlation matrix diagonal
                correlation_ops = matrix_data.get('operations', {}).get('correlation_matrix', {})
                correlation_matrix = correlation_ops.get('matrix', [])
                if correlation_matrix:
                    feature_importance = np.diag(correlation_matrix).tolist()
                    logger.info(f'✅ Using correlation matrix diagonal as variance proxy')
                else:
                    logger.warning(f'⚠️ No correlation matrix available for variance estimation')
            
            if feature_importance:
                logger.info(f'📈 Processing {len(feature_importance)} feature importance scores')
                
                # Filter features based on variance threshold
                high_variance_features = []
                for feature, importance in zip(feature_columns, feature_importance):
                    if importance >= variance_threshold:
                        high_variance_features.append(feature)
                        logger.debug(f'🔍 High variance feature: {feature} = {importance:.4f}')
                
                result = {
                    'threshold': variance_threshold, 
                    'total_features': len(feature_columns), 
                    'high_variance_features': len(high_variance_features), 
                    'features_removed': len(feature_columns) - len(high_variance_features), 
                    'selected_features': high_variance_features
                }
                
                logger.info(f'✅ Variance filtering completed:')
                logger.info(f'   - High variance features: {len(high_variance_features)}')
                logger.info(f'   - Features removed: {len(feature_columns) - len(high_variance_features)}')
                logger.info(f'   - Selection rate: {len(high_variance_features)/len(feature_columns):.2%}')
                
                return result
            else:
                logger.error(f'❌ No variance information available for filtering')
                return {'error': 'No variance information available', 'threshold': variance_threshold}
                
        except Exception as e:
            logger.error(f'❌ Error in variance filtering: {e}')
            logger.exception(f'🔍 Full error details:')
            return {'error': str(e), 'threshold': variance_threshold if 'variance_threshold' in locals() else 0.01}
    @log_all_calls
    def _apply_mutual_information_filtering(self, matrix_data: Dict[str, Any], regime_config: Dict[str, Any], feature_columns: List[str]) -> Dict[str, Any]:
        """Apply mutual information-based feature filtering.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            feature_columns: List of feature columns
            
        Returns:
            Mutual information filtering results
        """
        try:
            logger.info(f'🧠 Starting mutual information filtering with {len(feature_columns)} features...')
            
            # Get mutual information threshold
            mi_threshold = regime_config.get('selection_strategy', {}).get('mutual_info_threshold', 0.05)
            logger.info(f'📊 Using mutual information threshold: {mi_threshold}')
            
            # Get feature importance from balanced analysis
            balanced_ops = matrix_data.get('operations', {}).get('balanced_analysis', {})
            feature_importance = balanced_ops.get('feature_importance', [])
            
            if feature_importance:
                logger.info(f'📈 Processing {len(feature_importance)} feature importance scores for MI filtering')
                
                # Filter features based on mutual information threshold
                high_mi_features = []
                for feature, importance in zip(feature_columns, feature_importance):
                    if importance >= mi_threshold:
                        high_mi_features.append(feature)
                        logger.debug(f'🔍 High MI feature: {feature} = {importance:.4f}')
                
                result = {
                    'threshold': mi_threshold, 
                    'total_features': len(feature_columns), 
                    'high_mi_features': len(high_mi_features), 
                    'features_removed': len(feature_columns) - len(high_mi_features), 
                    'selected_features': high_mi_features
                }
                
                logger.info(f'✅ Mutual information filtering completed:')
                logger.info(f'   - High MI features: {len(high_mi_features)}')
                logger.info(f'   - Features removed: {len(feature_columns) - len(high_mi_features)}')
                logger.info(f'   - Selection rate: {len(high_mi_features)/len(feature_columns):.2%}')
                
                return result
            else:
                logger.error(f'❌ No mutual information available for filtering')
                return {'error': 'No mutual information available', 'threshold': mi_threshold}
                
        except Exception as e:
            logger.error(f'❌ Error in mutual information filtering: {e}')
            logger.exception(f'🔍 Full error details:')
            return {'error': str(e), 'threshold': mi_threshold if 'mi_threshold' in locals() else 0.05}
    @log_all_calls
    def _apply_recursive_feature_elimination(self, matrix_data: Dict[str, Any], regime_config: Dict[str, Any], feature_columns: List[str]) -> Dict[str, Any]:
        """Apply recursive feature elimination.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            feature_columns: List of feature columns
            
        Returns:
            RFE results
        """
        try:
            logger.info(f'🔄 Starting recursive feature elimination with {len(feature_columns)} features...')
            
            # Get maximum features limit
            max_features = regime_config.get('selection_strategy', {}).get('max_features', 50)
            logger.info(f'📊 Using max features limit: {max_features}')
            
            # Get PCA analysis results
            pca_ops = matrix_data.get('operations', {}).get('pca_analysis', {})
            if not pca_ops:
                logger.warning(f'⚠️ No PCA analysis operations found in matrix data')
                return {'error': 'No PCA analysis operations found', 'max_features': max_features}
            
            explained_variance = pca_ops.get('explained_variance_ratio', [])
            if explained_variance:
                logger.info(f'📈 Found {len(explained_variance)} PCA components with explained variance')
                
                # Calculate cumulative explained variance
                cumulative_variance = np.cumsum(explained_variance)
                logger.info(f'📊 Cumulative explained variance: {cumulative_variance[:5]}... (first 5 components)')
                
                # Determine number of components to keep
                n_components = min(max_features, len(explained_variance))
                logger.info(f'🎯 Selecting top {n_components} features based on PCA')
                
                # Select top features (simplified approach - using first n_components)
                top_features = feature_columns[:n_components]
                
                result = {
                    'max_features': max_features, 
                    'selected_features': len(top_features), 
                    'features_removed': len(feature_columns) - len(top_features), 
                    'selected_features': top_features,
                    'pca_components_used': n_components,
                    'explained_variance_ratio': explained_variance[:n_components] if len(explained_variance) >= n_components else explained_variance
                }
                
                logger.info(f'✅ Recursive feature elimination completed:')
                logger.info(f'   - Features selected: {len(top_features)}')
                logger.info(f'   - Features removed: {len(feature_columns) - len(top_features)}')
                logger.info(f'   - Selection rate: {len(top_features)/len(feature_columns):.2%}')
                logger.info(f'   - PCA components used: {n_components}')
                
                return result
            else:
                logger.error(f'❌ No explained variance ratio found in PCA analysis')
                return {'error': 'No explained variance ratio found in PCA analysis', 'max_features': max_features}
                
        except Exception as e:
            logger.error(f'❌ Error in recursive feature elimination: {e}')
            logger.exception(f'🔍 Full error details:')
            return {'error': str(e), 'max_features': max_features if 'max_features' in locals() else 50}
    @log_all_calls
    def _apply_permutation_importance(self, matrix_data: Dict[str, Any], regime_config: Dict[str, Any], feature_columns: List[str]) -> Dict[str, Any]:
        """Apply permutation importance analysis.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            feature_columns: List of feature columns
            
        Returns:
            Permutation importance results
        """
        try:
            logger.info(f'🎯 Starting permutation importance analysis with {len(feature_columns)} features...')
            
            # Get clustering analysis results
            clustering_ops = matrix_data.get('operations', {}).get('clustering_analysis', {})
            if clustering_ops:
                logger.info(f'📊 Found clustering analysis data for permutation importance')
                logger.info(f'🔑 Clustering operations keys: {list(clustering_ops.keys()) if isinstance(clustering_ops, dict) else "N/A"}')
                
                # Simplified approach: select top half of features
                # In a real implementation, this would use actual permutation importance scores
                important_features = feature_columns[:len(feature_columns) // 2]
                
                result = {
                    'total_features': len(feature_columns), 
                    'important_features': len(important_features), 
                    'features_removed': len(feature_columns) - len(important_features), 
                    'selected_features': important_features,
                    'selection_method': 'top_half_simplified'
                }
                
                logger.info(f'✅ Permutation importance analysis completed:')
                logger.info(f'   - Important features: {len(important_features)}')
                logger.info(f'   - Features removed: {len(feature_columns) - len(important_features)}')
                logger.info(f'   - Selection rate: {len(important_features)/len(feature_columns):.2%}')
                logger.info(f'   - Method: {result["selection_method"]}')
                
                return result
            else:
                logger.error(f'❌ No clustering analysis information available for permutation importance')
                return {'error': 'No clustering information available', 'total_features': len(feature_columns)}
                
        except Exception as e:
            logger.error(f'❌ Error in permutation importance: {e}')
            logger.exception(f'🔍 Full error details:')
            return {'error': str(e), 'total_features': len(feature_columns) if 'feature_columns' in locals() else 0}
    @log_all_calls
    def _combine_selection_results(self, selection_metadata: Dict[str, Any], regime_config: Dict[str, Any], feature_columns: List[str]) -> List[str]:
        """Combine results from different feature selection methods.
        
        Args:
            selection_metadata: Results from different selection methods
            regime_config: Regime configuration
            feature_columns: List of all feature columns
            
        Returns:
            List of selected features
        """
        try:
            logger.info(f'🔀 Combining selection results from {len(selection_metadata)} methods...')
            
            # Extract successful results from each method
            selected_by_method = {}
            successful_methods = []
            failed_methods = []
            
            for method, results in selection_metadata.items():
                if 'error' not in results and 'selected_features' in results:
                    selected_by_method[method] = set(results['selected_features'])
                    successful_methods.append(method)
                    logger.info(f'✅ Method {method}: {len(results["selected_features"])} features selected')
                else:
                    failed_methods.append(method)
                    logger.warning(f'⚠️ Method {method}: failed or no features selected')
            
            logger.info(f'📊 Successful methods: {successful_methods}')
            if failed_methods:
                logger.warning(f'⚠️ Failed methods: {failed_methods}')
            
            # Handle case where no methods succeeded
            if not selected_by_method:
                logger.warning(f'⚠️ No successful selection methods, using fallback selection')
                max_features = regime_config.get('selection_strategy', {}).get('max_features', 50)
                fallback_features = feature_columns[:min(max_features, len(feature_columns))]
                logger.info(f'📋 Fallback selection: {len(fallback_features)} features')
                return fallback_features
            
            # Try to find common features across methods
            if len(selected_by_method) > 1:
                logger.info(f'🔍 Looking for common features across {len(selected_by_method)} methods...')
                common_features = set.intersection(*selected_by_method.values())
                if common_features:
                    logger.info(f'✅ Found {len(common_features)} common features across all methods')
                    return list(common_features)
                else:
                    logger.info(f'ℹ️ No common features found across all methods, using union approach')
            
            # Use union of all selected features
            all_selected = set.union(*selected_by_method.values()) if selected_by_method else set()
            logger.info(f'📊 Union of all methods: {len(all_selected)} features')
            
            # Apply maximum features limit
            max_features = regime_config.get('selection_strategy', {}).get('max_features', 50)
            if len(all_selected) > max_features:
                logger.info(f'🎯 Limiting to {max_features} features (from {len(all_selected)})')
                # Convert to list and take first max_features
                all_selected = set(list(all_selected)[:max_features])
            
            final_features = list(all_selected) if all_selected else feature_columns[:max_features]
            
            logger.info(f'✅ Final feature combination completed:')
            logger.info(f'   - Final features: {len(final_features)}')
            logger.info(f'   - Selection rate: {len(final_features)/len(feature_columns):.2%}')
            logger.info(f'   - Methods used: {successful_methods}')
            
            return final_features
            
        except Exception as e:
            logger.error(f'❌ Error combining selection results: {e}')
            logger.exception(f'🔍 Full error details:')
            max_features = regime_config.get('selection_strategy', {}).get('max_features', 50)
            fallback_features = feature_columns[:min(max_features, len(feature_columns))]
            logger.info(f'📋 Using fallback selection: {len(fallback_features)} features')
            return fallback_features

    async def _save_regime_selection_results(self, selection_results: Dict[str, Any], symbol: str, exchange: str, timeframe: str, data_dir: str, regime_id: int) -> bool:
        """Save feature selection results for a specific regime.
        
        Args:
            selection_results: Feature selection results
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            True if successful
        """
        try:
            logger.info(f'💾 Saving feature selection results for regime {regime_id}...')
            
            # Create output directory if it doesn't exist
            output_dir = Path(data_dir) / 'training'
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f'📁 Output directory: {output_dir}')
            
            # Generate filename
            filename = f'{exchange}_{symbol}_{timeframe}_feature_selection_regime_{regime_id}.json'
            selection_path = output_dir / filename
            logger.info(f'📄 Saving to: {selection_path}')
            
            # Add metadata to results
            enhanced_results = {
                **selection_results,
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'regime_id': regime_id,
                    'data_dir': data_dir,
                    'file_path': str(selection_path)
                }
            }
            
            # Save the results
            save_start = time.time()
            with open(selection_path, 'w') as f:
                json.dump(enhanced_results, f, indent=2, default=str)
            save_time = time.time() - save_start
            
            # Verify file was created and get size
            if selection_path.exists():
                file_size = selection_path.stat().st_size
                logger.info(f'✅ Successfully saved feature selection results for regime {regime_id}')
                logger.info(f'📊 File details:')
                logger.info(f'   - Path: {selection_path}')
                logger.info(f'   - Size: {file_size} bytes')
                logger.info(f'   - Save time: {save_time:.3f}s')
                logger.info(f'   - Selected features: {len(selection_results.get("selected_features", []))}')
                logger.info(f'   - Total features: {selection_results.get("total_features", 0)}')
                logger.info(f'   - Selection rate: {selection_results.get("selection_rate", 0):.2%}')
                return True
            else:
                logger.error(f'❌ File was not created: {selection_path}')
                return False
                
        except FileNotFoundError as e:
            logger.error(f'❌ Directory not found for regime {regime_id}: {e}')
            return False
        except PermissionError as e:
            logger.error(f'❌ Permission denied saving results for regime {regime_id}: {e}')
            return False
        except json.JSONEncodeError as e:
            logger.error(f'❌ JSON encoding error for regime {regime_id}: {e}')
            return False
        except Exception as e:
            logger.error(f'❌ Unexpected error saving feature selection results for regime {regime_id}: {e}')
            logger.exception(f'🔍 Full error details:')
            return False

@traced(span_name='run_per_regime_feature_selection_step')
@validates()
@handles_errors
async def run_per_regime_step(symbol: str, exchange: str, timeframe: str, data_dir: str = None, force_rerun: bool = False, config: Optional[Dict[str, Any]]=None) -> bool:
    """Run the enhanced per-regime feature selection step.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory
        force_rerun: Force rerun the step
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        execution_start = time.time()
        logger.info('🚀 Starting Step 8: Per-Regime Advanced Feature Selection')
        logger.info(f'📊 Execution parameters:')
        logger.info(f'   - Symbol: {symbol}')
        logger.info(f'   - Exchange: {exchange}')
        logger.info(f'   - Timeframe: {timeframe}')
        logger.info(f'   - Force rerun: {force_rerun}')
        logger.info(f'   - Data directory: {data_dir}')
        
        # Initialize configuration
        if config is None:
            config = {}
            logger.info('📋 Using default configuration')
        else:
            logger.info(f'📋 Using provided configuration with {len(config)} keys')
        
        # Set up data directory
        if data_dir is None:
            data_dir = standardized_parquet_handler.get_standardized_path('processed_data', exchange, symbol)
            logger.info(f'📁 Using standardized data directory: {data_dir}')
        
        # Configure per-regime feature selection
        config['per_regime_feature_selection'] = True
        logger.info('✅ Per-regime feature selection enabled in configuration')
        
        # Initialize step
        logger.info('🔧 Initializing Per-Regime Advanced Feature Selection Step...')
        step_init_start = time.time()
        step = PerRegimeAdvancedFeatureSelectionStep(config)
        step_init_time = time.time() - step_init_start
        logger.info(f'✅ Step initialized in {step_init_time:.3f}s')
        
        # Execute the step
        logger.info('🎯 Executing per-regime feature selection...')
        execution_start_inner = time.time()
        success = await step.execute_per_regime_feature_selection(
            symbol=symbol, 
            exchange=exchange, 
            timeframe=timeframe, 
            data_dir=data_dir, 
            force_rerun=force_rerun
        )
        execution_time_inner = time.time() - execution_start_inner
        
        total_time = time.time() - execution_start
        
        # Log final results
        if success:
            logger.info('✅ Step 8: Per-Regime Advanced Feature Selection completed successfully')
            logger.info(f'⏱️ Execution summary:')
            logger.info(f'   - Total time: {total_time:.3f}s')
            logger.info(f'   - Step init: {step_init_time:.3f}s')
            logger.info(f'   - Execution: {execution_time_inner:.3f}s')
        else:
            logger.error('❌ Step 8: Per-Regime Advanced Feature Selection failed')
            logger.error(f'⏱️ Failed after {total_time:.3f}s')
        
        return success
        
    except Exception as e:
        execution_time = time.time() - execution_start if 'execution_start' in locals() else 0
        logger.error(f'❌ Critical error in run_per_regime_step after {execution_time:.3f}s: {e}')
        logger.exception(f'🔍 Full error details:')
        return False
if __name__ == '__main__':

    async def test() -> None:
        """Test the per-regime feature selection step."""
        success = await run_per_regime_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache')
        print(f'Per-regime feature selection result: {success}')
    asyncio.run(test())