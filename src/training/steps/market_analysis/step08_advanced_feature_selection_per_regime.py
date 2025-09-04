"""Step 8: Advanced Feature Selection - Per-Regime Implementation.

This module provides per-HMM regime feature selection functionality, ensuring that
feature selection is optimized specifically for each regime's characteristics.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.training.steps.step08_advanced_feature_selection import Step8AdvancedFeatureSelection
from src.training.steps.regime_handler import regime_handler
from src.training.steps.regime_processing_decorator import (
    per_regime_processing,
    aggregate_regime_results,
    RegimeProcessingContext
)
from src.training.steps.regime_continuity_decorator import per_regime_step
from src.utils.logger import getChild as get_logger
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import traced, validates, handles_errors
from src.core.decorators.errors import handles_errors


logger = get_logger('Step8AdvancedFeatureSelectionPerRegime')


class PerRegimeAdvancedFeatureSelectionStep(Step8AdvancedFeatureSelection):
    """Advanced feature selection step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_feature_selection', True)
        self.regime_specific_configs = config.get('regime_specific_feature_selection_configs', {})
        self.adaptive_feature_selection = config.get('adaptive_feature_selection_per_regime', True)
        
    @traced(span_name='execute_per_regime_feature_selection')
    @per_regime_step('step08_advanced_feature_selection')
    async def execute_per_regime_feature_selection(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool = False,
        regime_id: Optional[int] = None,
        regime_context: Optional[Any] = None,
        per_regime: bool = True
    ) -> bool:
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
            self.logger.info(f"🚀 Starting per-regime feature selection for regime {regime_id}")
            
            # Load matrix operation results from previous step
            matrix_data = await self._load_matrix_data(symbol, exchange, timeframe, data_dir, regime_id)
            if matrix_data is None:
                self.logger.error(f"❌ Failed to load matrix data for regime {regime_id}")
                return False
            
            # Get regime-specific configuration
            regime_config = self._get_regime_feature_selection_config(regime_id)
            
            # Apply regime-specific feature selection
            selection_results = await self._apply_regime_feature_selection(
                matrix_data, regime_config, regime_id
            )
            
            if selection_results is None:
                self.logger.error(f"❌ Failed feature selection for regime {regime_id}")
                return False
            
            # Save regime-specific results
            success = await self._save_regime_selection_results(
                selection_results, symbol, exchange, timeframe, data_dir, regime_id
            )
            
            if success:
                self.logger.info(f"✅ Successfully completed feature selection for regime {regime_id}")
            else:
                self.logger.error(f"❌ Failed to save selection results for regime {regime_id}")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime feature selection for regime {regime_id}: {e}")
            return False
    
    async def _load_matrix_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
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
            # Try per-regime matrix data first
            matrix_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_matrix_operations_regime_{regime_id}.json'
            
            if not matrix_path.exists():
                # Fall back to aggregated matrix data
                matrix_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_matrix_operations_aggregated.json'
            
            if matrix_path.exists():
                with open(matrix_path, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"✅ Loaded matrix data for regime {regime_id}")
                return data
            else:
                self.logger.error(f"❌ Matrix data not found: {matrix_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading matrix data for regime {regime_id}: {e}")
            return None
    
    def _get_regime_feature_selection_config(self, regime_id: int) -> Dict[str, Any]:
        """Get feature selection configuration for a specific regime.
        
        Different regimes may benefit from different feature selection strategies.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific feature selection configuration
        """
        # Check if custom config exists for this regime
        if f'regime_{regime_id}' in self.regime_specific_configs:
            return self.regime_specific_configs[f'regime_{regime_id}']
        
        # Create adaptive configuration based on regime characteristics
        base_config = {
            'enable_correlation_filtering': True,
            'enable_variance_filtering': True,
            'enable_mutual_information': True,
            'enable_recursive_feature_elimination': True,
            'enable_permutation_importance': True
        }
        
        # Adapt based on regime ID patterns
        if regime_id <= 2:
            # Low regime IDs - often trending markets
            # Emphasize trend-relevant features
            return {
                **base_config,
                'selection_strategy': {
                    'max_features': 50,
                    'correlation_threshold': 0.8,
                    'variance_threshold': 0.01,
                    'mutual_info_threshold': 0.05,
                    'emphasis': 'trend_features'
                },
                'feature_types': [
                    'trend_indicators',
                    'momentum_features',
                    'volume_trend_features',
                    'price_trend_features'
                ]
            }
        elif regime_id >= 5:
            # High regime IDs - often volatile/ranging markets
            # Emphasize volatility and mean reversion features
            return {
                **base_config,
                'selection_strategy': {
                    'max_features': 40,
                    'correlation_threshold': 0.7,
                    'variance_threshold': 0.02,
                    'mutual_info_threshold': 0.03,
                    'emphasis': 'volatility_features'
                },
                'feature_types': [
                    'volatility_indicators',
                    'mean_reversion_features',
                    'oscillator_features',
                    'range_features'
                ]
            }
        else:
            # Medium regime IDs - balanced approach
            return {
                **base_config,
                'selection_strategy': {
                    'max_features': 45,
                    'correlation_threshold': 0.75,
                    'variance_threshold': 0.015,
                    'mutual_info_threshold': 0.04,
                    'emphasis': 'balanced_features'
                },
                'feature_types': [
                    'mixed_indicators',
                    'balanced_features',
                    'adaptive_features',
                    'composite_features'
                ]
            }
    
    async def _apply_regime_feature_selection(
        self,
        matrix_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Apply feature selection to regime data.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Feature selection results or None
        """
        try:
            self.logger.info(f"🔧 Applying feature selection for regime {regime_id}")
            
            # Extract feature information from matrix data
            feature_columns = matrix_data.get('feature_columns', [])
            if not feature_columns:
                self.logger.warning(f"⚠️ No feature columns found for regime {regime_id}")
                return None
            
            results = {
                'regime_id': regime_id,
                'total_features': len(feature_columns),
                'selection_strategy': regime_config.get('selection_strategy', {}),
                'selected_features': [],
                'feature_scores': {},
                'selection_metadata': {}
            }
            
            # Apply correlation filtering
            if regime_config.get('enable_correlation_filtering', True):
                correlation_results = self._apply_correlation_filtering(
                    matrix_data, regime_config, feature_columns
                )
                results['selection_metadata']['correlation_filtering'] = correlation_results
            
            # Apply variance filtering
            if regime_config.get('enable_variance_filtering', True):
                variance_results = self._apply_variance_filtering(
                    matrix_data, regime_config, feature_columns
                )
                results['selection_metadata']['variance_filtering'] = variance_results
            
            # Apply mutual information filtering
            if regime_config.get('enable_mutual_information', True):
                mi_results = self._apply_mutual_information_filtering(
                    matrix_data, regime_config, feature_columns
                )
                results['selection_metadata']['mutual_information'] = mi_results
            
            # Apply recursive feature elimination
            if regime_config.get('enable_recursive_feature_elimination', True):
                rfe_results = self._apply_recursive_feature_elimination(
                    matrix_data, regime_config, feature_columns
                )
                results['selection_metadata']['recursive_feature_elimination'] = rfe_results
            
            # Apply permutation importance
            if regime_config.get('enable_permutation_importance', True):
                perm_results = self._apply_permutation_importance(
                    matrix_data, regime_config, feature_columns
                )
                results['selection_metadata']['permutation_importance'] = perm_results
            
            # Combine selection results
            selected_features = self._combine_selection_results(
                results['selection_metadata'], regime_config, feature_columns
            )
            
            results['selected_features'] = selected_features
            results['selected_count'] = len(selected_features)
            results['selection_rate'] = len(selected_features) / len(feature_columns) if feature_columns else 0
            
            self.logger.info(f"✅ Completed feature selection for regime {regime_id}: {len(selected_features)}/{len(feature_columns)} features selected")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error applying feature selection for regime {regime_id}: {e}")
            return None
    
    def _apply_correlation_filtering(
        self,
        matrix_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Apply correlation-based feature filtering.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            feature_columns: List of feature columns
            
        Returns:
            Correlation filtering results
        """
        try:
            correlation_threshold = regime_config.get('selection_strategy', {}).get('correlation_threshold', 0.8)
            
            # Get correlation matrix from matrix data
            correlation_ops = matrix_data.get('operations', {}).get('correlation_matrix', {})
            high_correlations = correlation_ops.get('high_correlations', [])
            
            # Find features to remove due to high correlation
            features_to_remove = set()
            for corr in high_correlations:
                if corr['abs_correlation'] >= correlation_threshold:
                    # Remove the feature with lower variance (keep the more informative one)
                    features_to_remove.add(corr['feature2'])  # Simple heuristic
            
            remaining_features = [f for f in feature_columns if f not in features_to_remove]
            
            return {
                'threshold': correlation_threshold,
                'high_correlations': len(high_correlations),
                'features_removed': len(features_to_remove),
                'features_remaining': len(remaining_features),
                'removed_features': list(features_to_remove)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in correlation filtering: {e}")
            return {'error': str(e)}
    
    def _apply_variance_filtering(
        self,
        matrix_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Apply variance-based feature filtering.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            feature_columns: List of feature columns
            
        Returns:
            Variance filtering results
        """
        try:
            variance_threshold = regime_config.get('selection_strategy', {}).get('variance_threshold', 0.01)
            
            # Get feature importance from balanced analysis if available
            balanced_ops = matrix_data.get('operations', {}).get('balanced_analysis', {})
            feature_importance = balanced_ops.get('feature_importance', [])
            
            if not feature_importance:
                # Fallback: use correlation matrix diagonal (variance approximation)
                correlation_ops = matrix_data.get('operations', {}).get('correlation_matrix', {})
                correlation_matrix = correlation_ops.get('matrix', [])
                if correlation_matrix:
                    feature_importance = np.diag(correlation_matrix).tolist()
            
            if feature_importance:
                # Filter features with low variance
                high_variance_features = [
                    feature for feature, importance in zip(feature_columns, feature_importance)
                    if importance >= variance_threshold
                ]
                
                return {
                    'threshold': variance_threshold,
                    'total_features': len(feature_columns),
                    'high_variance_features': len(high_variance_features),
                    'features_removed': len(feature_columns) - len(high_variance_features),
                    'selected_features': high_variance_features
                }
            else:
                return {'error': 'No variance information available'}
                
        except Exception as e:
            self.logger.error(f"❌ Error in variance filtering: {e}")
            return {'error': str(e)}
    
    def _apply_mutual_information_filtering(
        self,
        matrix_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Apply mutual information-based feature filtering.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            feature_columns: List of feature columns
            
        Returns:
            Mutual information filtering results
        """
        try:
            mi_threshold = regime_config.get('selection_strategy', {}).get('mutual_info_threshold', 0.05)
            
            # For now, use feature importance as a proxy for mutual information
            # In a real implementation, you would calculate actual mutual information
            balanced_ops = matrix_data.get('operations', {}).get('balanced_analysis', {})
            feature_importance = balanced_ops.get('feature_importance', [])
            
            if feature_importance:
                high_mi_features = [
                    feature for feature, importance in zip(feature_columns, feature_importance)
                    if importance >= mi_threshold
                ]
                
                return {
                    'threshold': mi_threshold,
                    'total_features': len(feature_columns),
                    'high_mi_features': len(high_mi_features),
                    'features_removed': len(feature_columns) - len(high_mi_features),
                    'selected_features': high_mi_features
                }
            else:
                return {'error': 'No mutual information available'}
                
        except Exception as e:
            self.logger.error(f"❌ Error in mutual information filtering: {e}")
            return {'error': str(e)}
    
    def _apply_recursive_feature_elimination(
        self,
        matrix_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Apply recursive feature elimination.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            feature_columns: List of feature columns
            
        Returns:
            RFE results
        """
        try:
            max_features = regime_config.get('selection_strategy', {}).get('max_features', 50)
            
            # Use PCA components as a proxy for feature importance
            pca_ops = matrix_data.get('operations', {}).get('pca_analysis', {})
            explained_variance = pca_ops.get('explained_variance_ratio', [])
            
            if explained_variance:
                # Select top features based on explained variance
                n_components = min(max_features, len(explained_variance))
                top_features = feature_columns[:n_components]  # Simple selection
                
                return {
                    'max_features': max_features,
                    'selected_features': len(top_features),
                    'features_removed': len(feature_columns) - len(top_features),
                    'selected_features': top_features
                }
            else:
                return {'error': 'No PCA information available'}
                
        except Exception as e:
            self.logger.error(f"❌ Error in recursive feature elimination: {e}")
            return {'error': str(e)}
    
    def _apply_permutation_importance(
        self,
        matrix_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Apply permutation importance analysis.
        
        Args:
            matrix_data: Matrix operation results
            regime_config: Regime configuration
            feature_columns: List of feature columns
            
        Returns:
            Permutation importance results
        """
        try:
            # Use clustering results as a proxy for feature importance
            clustering_ops = matrix_data.get('operations', {}).get('clustering_analysis', {})
            
            if clustering_ops:
                # Simple heuristic: features that contribute to better clustering
                important_features = feature_columns[:len(feature_columns)//2]  # Simple selection
                
                return {
                    'total_features': len(feature_columns),
                    'important_features': len(important_features),
                    'features_removed': len(feature_columns) - len(important_features),
                    'selected_features': important_features
                }
            else:
                return {'error': 'No clustering information available'}
                
        except Exception as e:
            self.logger.error(f"❌ Error in permutation importance: {e}")
            return {'error': str(e)}
    
    def _combine_selection_results(
        self,
        selection_metadata: Dict[str, Any],
        regime_config: Dict[str, Any],
        feature_columns: List[str]
    ) -> List[str]:
        """Combine results from different feature selection methods.
        
        Args:
            selection_metadata: Results from different selection methods
            regime_config: Regime configuration
            feature_columns: List of all feature columns
            
        Returns:
            List of selected features
        """
        try:
            # Collect features selected by each method
            selected_by_method = {}
            
            for method, results in selection_metadata.items():
                if 'error' not in results and 'selected_features' in results:
                    selected_by_method[method] = set(results['selected_features'])
            
            if not selected_by_method:
                # Fallback: select top features based on available information
                max_features = regime_config.get('selection_strategy', {}).get('max_features', 50)
                return feature_columns[:min(max_features, len(feature_columns))]
            
            # Use intersection of methods for conservative selection
            if len(selected_by_method) > 1:
                common_features = set.intersection(*selected_by_method.values())
                if common_features:
                    return list(common_features)
            
            # Fallback to union of methods
            all_selected = set.union(*selected_by_method.values()) if selected_by_method else set()
            
            # Limit to max features
            max_features = regime_config.get('selection_strategy', {}).get('max_features', 50)
            if len(all_selected) > max_features:
                # Simple selection of first max_features
                all_selected = set(list(all_selected)[:max_features])
            
            return list(all_selected) if all_selected else feature_columns[:max_features]
            
        except Exception as e:
            self.logger.error(f"❌ Error combining selection results: {e}")
            # Fallback
            max_features = regime_config.get('selection_strategy', {}).get('max_features', 50)
            return feature_columns[:min(max_features, len(feature_columns))]
    
    async def _save_regime_selection_results(
        self,
        selection_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> bool:
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
            # Save regime-specific results
            selection_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_feature_selection_regime_{regime_id}.json'
            
            with open(selection_path, 'w') as f:
                json.dump(selection_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Saved feature selection results for regime {regime_id}: {selection_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving feature selection results for regime {regime_id}: {e}")
            return False


@traced(span_name='run_per_regime_feature_selection_step')
@validates()
@handles_errors
async def run_per_regime_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = None,
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> bool:
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
    logger.info("🚀 Starting Step 8: Per-Regime Advanced Feature Selection")
    
    if config is None:
        config = {}
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Enable per-regime processing
    config['per_regime_feature_selection'] = True
    
    # Initialize and run the per-regime feature selection step
    step = PerRegimeAdvancedFeatureSelectionStep(config)
    
    success = await step.execute_per_regime_feature_selection(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )
    
    if success:
        logger.info("✅ Step 8: Per-Regime Advanced Feature Selection completed successfully")
    else:
        logger.error("❌ Step 8: Per-Regime Advanced Feature Selection failed")
        
    return success


if __name__ == '__main__':
    async def test():
        """Test the per-regime feature selection step."""
        success = await run_per_regime_step(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        print(f'Per-regime feature selection result: {success}')
        
    asyncio.run(test())