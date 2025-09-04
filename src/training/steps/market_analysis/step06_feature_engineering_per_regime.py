"""Enhanced Step 6: Per-Regime Feature Engineering.

This module provides per-HMM regime feature engineering functionality, ensuring that
features are engineered specifically for each regime's characteristics.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np

from src.training.steps.step06_feature_engineering import FeatureInteractionEngine
from src.training.steps.regime_handler import regime_handler
from src.training.steps.regime_processing_decorator import (
    per_regime_processing,
    aggregate_regime_results,
    RegimeProcessingContext
)
from src.utils.logger import getChild as get_logger
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import traced, validates, handles_errors


logger = get_logger('Step6FeatureEngineeringPerRegime')


class PerRegimeFeatureEngineeringStep(FeatureInteractionEngine):
    """Enhanced feature engineering step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_feature_engineering', True)
        self.regime_specific_features = config.get('regime_specific_features', {})
        self.adaptive_lookback = config.get('adaptive_lookback_per_regime', True)
        
        # Force regime-specific optimization for per-regime processing
        step6_config = config.get('step06_feature_engineering', {})
        step6_config['force_regime_specific_periods'] = True
        config['step06_feature_engineering'] = step6_config
        
        # Update the parent class configuration
        self.config = config
        self.force_regime_specific_periods = True
        
        self.logger.info("🎯 Per-regime feature engineering initialized with regime-specific optimization enabled")
        
    @traced(span_name='execute_per_regime_feature_engineering')
    async def execute_per_regime_feature_engineering(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool = False
    ) -> bool:
        """Execute feature engineering on a per-regime basis.
        
        Each regime may have different market dynamics, so features should be
        engineered specifically for each regime's characteristics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            
        Returns:
            Success status
        """
        try:
            self.logger.info("🚀 Starting per-regime feature engineering process")
            
            # Load labeled data from previous step
            labeled_data = await self._load_labeled_data(symbol, exchange, timeframe, data_dir)
            if labeled_data is None:
                self.logger.error("❌ Failed to load labeled data")
                return False
            
            # Use regime processing context
            async with RegimeProcessingContext(symbol, exchange, timeframe, data_dir) as ctx:
                if ctx.regime_data is None:
                    self.logger.error("❌ Failed to load regime data")
                    return False
                    
                self.logger.info(f"📊 Engineering features for {len(ctx.regime_ids)} regimes")
                
                # Process each regime
                regime_results = {}
                regime_feature_info = {}
                
                for regime_id in ctx.regime_ids:
                    self.logger.info(f"🔄 Processing regime {regime_id}")
                    
                    # Get regime-specific configuration
                    regime_config = self._get_regime_feature_config(regime_id)
                    
                    # Process this regime
                    result, feature_info = await self._engineer_features_single_regime(
                        ctx=ctx,
                        regime_id=regime_id,
                        labeled_data=labeled_data,
                        regime_config=regime_config
                    )
                    
                    if result is not None:
                        regime_results[regime_id] = result
                        regime_feature_info[regime_id] = feature_info
                        self.logger.info(f"✅ Successfully engineered features for regime {regime_id}")
                    else:
                        self.logger.error(f"❌ Failed to engineer features for regime {regime_id}")
                
                # Save per-regime results
                success = await regime_handler.save_regime_results(
                    results=regime_results,
                    step_name='step06_feature_engineering',
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data_dir=data_dir,
                    result_type='feature_engineered_data'
                )
                
                if success:
                    # Save feature metadata
                    await self._save_regime_feature_metadata(
                        regime_feature_info,
                        symbol,
                        exchange,
                        timeframe,
                        data_dir
                    )
                    
                    # Aggregate results for unified output
                    aggregated = self._aggregate_regime_features(regime_results)
                    
                    # Save aggregated results
                    output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_features_per_regime.parquet'
                    aggregated.to_parquet(output_path, index=False)
                    self.logger.info(f"✅ Saved aggregated feature data: {output_path}")
                    
                    # Log feature statistics
                    self._log_feature_statistics(aggregated, regime_feature_info)
                    
                return success
                
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime feature engineering: {e}")
            return False
    
    async def _engineer_features_single_regime(
        self,
        ctx: RegimeProcessingContext,
        regime_id: int,
        labeled_data: pd.DataFrame,
        regime_config: Dict[str, Any]
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Engineer features for a single regime.
        
        Args:
            ctx: Regime processing context
            regime_id: Regime ID to process
            labeled_data: Labeled data from previous step
            regime_config: Configuration for this regime
            
        Returns:
            Tuple of (feature engineered DataFrame, feature metadata)
        """
        try:
            # Get regime data with context
            regime_data = ctx.get_regime_data(regime_id, preserve_context=True)
            
            if regime_data.empty:
                self.logger.warning(f"⚠️ No data for regime {regime_id}")
                return None, None
            
            # Merge with labeled data
            regime_labeled = pd.merge(
                regime_data,
                labeled_data[['timestamp', 'label', 'label_type']],
                on='timestamp',
                how='left'
            )
            
            # Remove context indicator
            if 'is_regime_context' in regime_labeled.columns:
                context_mask = regime_labeled['is_regime_context']
                regime_labeled = regime_labeled.drop(columns=['is_regime_context'])
            else:
                context_mask = pd.Series(False, index=regime_labeled.index)
            
            # Optimize lookback periods for this regime if adaptive
            if self.adaptive_lookback:
                self.logger.info(f"🔍 Optimizing lookback periods for regime {regime_id}")
                
                # Only use non-context rows for optimization
                optimization_data = regime_labeled[~context_mask]
                
                if len(optimization_data) > 100:  # Need sufficient data
                    target = optimization_data['label']
                    
                    # Create regime-specific series for optimization
                    regime_series = pd.Series(regime_id, index=optimization_data.index)
                    
                    # Perform regime-specific optimization
                    optimization_results = await self.optimize_lookback_periods(
                        optimization_data,
                        target,
                        regimes=regime_series
                    )
                    
                    # Validate and process optimization results
                    if optimization_results.get('status') == 'optimized':
                        self.logger.info(f"✅ Regime {regime_id} optimization successful")
                        
                        # Extract regime-specific periods if available
                        regime_specific_periods = optimization_results.get('optimization_results', {}).get('regime_specific_periods', {})
                        regime_key = f'regime_{regime_id}'
                        
                        if regime_key in regime_specific_periods:
                            regime_periods = regime_specific_periods[regime_key]
                            regime_config['optimized_periods'] = regime_periods
                            self.logger.info(f"📊 Regime {regime_id} specific periods: {list(regime_periods.keys())}")
                        else:
                            # Fall back to global optimized periods
                            global_periods = optimization_results.get('periods', {})
                            regime_config['optimized_periods'] = global_periods
                            self.logger.info(f"📊 Using global optimized periods for regime {regime_id}")
                        
                        # Update interaction patterns with regime-specific periods
                        self._update_regime_interaction_patterns(regime_config, regime_id)
                        
                        # Validate the optimization results
                        validation_passed = self._validate_regime_optimization(
                            regime_id, optimization_results, regime_config
                        )
                        
                        if not validation_passed:
                            self.logger.warning(f"⚠️ Regime {regime_id} optimization validation failed, but continuing")
                        
                    elif optimization_results.get('status') == 'fallback':
                        self.logger.warning(f"⚠️ Regime {regime_id} using fallback periods")
                        regime_config['optimized_periods'] = optimization_results.get('periods', {})
                    else:
                        self.logger.error(f"❌ Regime {regime_id} optimization failed")
                        regime_config['optimized_periods'] = {}
                else:
                    self.logger.warning(f"⚠️ Insufficient data for regime {regime_id} optimization ({len(optimization_data)} rows)")
                    regime_config['optimized_periods'] = {}
            
            # Apply feature engineering
            features_df = await self._apply_feature_engineering(regime_labeled, regime_config)
            
            if features_df is None:
                return None, None
            
            # Add regime ID to features
            features_df['feature_regime_id'] = regime_id
            
            # Create feature metadata
            feature_info = {
                'regime_id': regime_id,
                'num_features': len([c for c in features_df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]),
                'feature_names': [c for c in features_df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']],
                'regime_config': regime_config,
                'data_shape': features_df.shape,
                'context_rows': int(context_mask.sum()) if context_mask is not None else 0,
                'optimization_info': {
                    'adaptive_lookback_enabled': self.adaptive_lookback,
                    'optimized_periods': regime_config.get('optimized_periods', {}),
                    'optimization_priority': regime_config.get('optimization_priority', 'unknown'),
                    'emphasis': regime_config.get('emphasis', 'unknown')
                }
            }
            
            return features_df, feature_info
            
        except Exception as e:
            self.logger.error(f"❌ Error engineering features for regime {regime_id}: {e}")
            return None, None
    
    def _get_regime_feature_config(self, regime_id: int) -> Dict[str, Any]:
        """Get feature engineering configuration for a specific regime.
        
        Different regimes may benefit from different feature sets and parameters.
        This method creates regime-specific configurations that will be used
        to optimize lookback periods and feature interactions.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific feature configuration
        """
        # Check if custom config exists for this regime
        if f'regime_{regime_id}' in self.regime_specific_features:
            custom_config = self.regime_specific_features[f'regime_{regime_id}']
            self.logger.info(f"📋 Using custom configuration for regime {regime_id}")
            return custom_config
        
        # Create adaptive configuration based on regime characteristics
        base_config = {
            'enable_technical_indicators': True,
            'enable_price_features': True,
            'enable_volume_features': True,
            'enable_volatility_features': True,
            'enable_microstructure_features': True,
            'force_regime_specific_periods': True,  # Ensure per-regime optimization
            'regime_id': regime_id
        }
        
        # Adapt based on regime ID patterns and market characteristics
        if regime_id <= 2:
            # Low regime IDs - often trending markets
            # Emphasize trend-following indicators with longer lookbacks
            config = {
                **base_config,
                'lookback_periods': [10, 20, 50, 100, 200],
                'emphasis': 'trend',
                'additional_features': [
                    'SMA_cross_features',
                    'EMA_ribbon',
                    'ADX_features',
                    'trend_strength'
                ],
                'interaction_patterns': {
                    'trend_momentum': {
                        'features': ['SMA_20', 'SMA_100', 'EMA_21', 'ADX_14'],
                        'weight': 2.0,
                        'enabled': True
                    },
                    'trend_volume': {
                        'features': ['OBV_20', 'Volume_Ratio', 'SMA_20', 'ATR_14'],
                        'weight': 1.8,
                        'enabled': True
                    }
                },
                'optimization_priority': 'trend_strength'
            }
            self.logger.info(f"📈 Configured regime {regime_id} for trending markets")
            
        elif regime_id >= 5:
            # High regime IDs - often volatile/ranging markets
            # Emphasize mean-reversion and volatility indicators with shorter lookbacks
            config = {
                **base_config,
                'lookback_periods': [5, 10, 20, 30],
                'emphasis': 'mean_reversion',
                'additional_features': [
                    'RSI_divergence',
                    'Bollinger_bands_features',
                    'ATR_bands',
                    'volatility_cones'
                ],
                'interaction_patterns': {
                    'mean_reversion': {
                        'features': ['RSI_14', 'BB_Position_20', 'Williams_R_14', 'CCI_20'],
                        'weight': 2.2,
                        'enabled': True
                    },
                    'volatility_regime': {
                        'features': ['ATR_14', 'BB_Squeeze_20', 'Volatility', 'Volume_Ratio'],
                        'weight': 1.9,
                        'enabled': True
                    }
                },
                'optimization_priority': 'volatility_capture'
            }
            self.logger.info(f"📊 Configured regime {regime_id} for volatile/ranging markets")
            
        else:
            # Medium regime IDs - balanced approach
            config = {
                **base_config,
                'lookback_periods': [7, 14, 30, 60],
                'emphasis': 'balanced',
                'additional_features': [
                    'momentum_features',
                    'volume_profile',
                    'market_microstructure'
                ],
                'interaction_patterns': {
                    'momentum_volume': {
                        'features': ['RSI_14', 'MACD_12_26', 'OBV_20', 'Volume_Ratio'],
                        'weight': 1.6,
                        'enabled': True
                    },
                    'oscillator_trend': {
                        'features': ['RSI_14', 'Williams_R_14', 'CCI_20', 'EMA_21'],
                        'weight': 1.4,
                        'enabled': True
                    }
                },
                'optimization_priority': 'balanced_performance'
            }
            self.logger.info(f"⚖️ Configured regime {regime_id} for balanced approach")
        
        return config
    
    def _update_regime_interaction_patterns(self, regime_config: Dict[str, Any], regime_id: int) -> None:
        """Update interaction patterns with regime-specific optimized periods.
        
        Args:
            regime_config: Regime configuration dictionary
            regime_id: Regime ID
        """
        try:
            optimized_periods = regime_config.get('optimized_periods', {})
            if not optimized_periods:
                self.logger.warning(f"⚠️ No optimized periods available for regime {regime_id}")
                return
            
            # Update interaction patterns with optimized periods
            interaction_patterns = regime_config.get('interaction_patterns', {})
            
            for pattern_name, pattern_config in interaction_patterns.items():
                updated_features = []
                for feature in pattern_config.get('features', []):
                    # Extract base indicator name
                    base_indicator = feature.split('_')[0]
                    
                    # Check if we have optimized periods for this indicator
                    if base_indicator in optimized_periods:
                        # Use the first optimized period
                        optimized_period = optimized_periods[base_indicator].get('selected_periods', [None])[0]
                        if optimized_period:
                            # Create feature name with optimized period
                            if '_' in feature:
                                parts = feature.split('_')
                                parts[1] = str(optimized_period)
                                updated_feature = '_'.join(parts)
                            else:
                                updated_feature = f"{base_indicator}_{optimized_period}"
                            updated_features.append(updated_feature)
                            self.logger.debug(f"🔄 Updated {feature} -> {updated_feature} for regime {regime_id}")
                        else:
                            updated_features.append(feature)
                    else:
                        updated_features.append(feature)
                
                # Update the pattern with optimized features
                pattern_config['features'] = updated_features
            
            regime_config['interaction_patterns'] = interaction_patterns
            self.logger.info(f"✅ Updated interaction patterns for regime {regime_id} with optimized periods")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating interaction patterns for regime {regime_id}: {e}")
    
    async def _apply_feature_engineering(
        self,
        regime_data: pd.DataFrame,
        regime_config: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Apply feature engineering to regime data.
        
        Args:
            regime_data: Regime-specific data
            regime_config: Regime configuration
            
        Returns:
            Feature engineered DataFrame or None
        """
        try:
            self.logger.info(f"🔧 Applying feature engineering for regime {regime_config.get('regime_id', 'unknown')}")
            
            # Extract technical indicators
            technical_features = self.extract_optimal_technical_indicators(regime_data)
            
            if technical_features.empty:
                self.logger.warning("⚠️ No technical features extracted")
                return None
            
            # Merge with original data
            features_df = pd.concat([regime_data, technical_features], axis=1)
            
            # Apply regime-specific interaction patterns if available
            interaction_patterns = regime_config.get('interaction_patterns', {})
            if interaction_patterns:
                self.logger.info(f"🔄 Applying {len(interaction_patterns)} interaction patterns")
                
                # Get feature names for interactions
                feature_names = [col for col in features_df.columns 
                               if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']]
                
                # Create interaction features
                interaction_features = self.extract_interaction_features(
                    features_df[feature_names].values,
                    feature_names,
                    regime_data
                )
                
                if interaction_features is not None and interaction_features.size > 0:
                    # Create interaction feature names
                    interaction_names = [f"interaction_{i}" for i in range(interaction_features.shape[1])]
                    interaction_df = pd.DataFrame(
                        interaction_features,
                        index=features_df.index,
                        columns=interaction_names
                    )
                    
                    # Merge interaction features
                    features_df = pd.concat([features_df, interaction_df], axis=1)
                    self.logger.info(f"✅ Added {len(interaction_names)} interaction features")
            
            # Add regime-specific metadata
            features_df['regime_emphasis'] = regime_config.get('emphasis', 'unknown')
            features_df['optimization_priority'] = regime_config.get('optimization_priority', 'unknown')
            
            self.logger.info(f"✅ Feature engineering completed: {features_df.shape[1]} total features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ Error applying feature engineering: {e}")
            return None
    
    def _validate_regime_optimization(
        self,
        regime_id: int,
        optimization_results: Dict[str, Any],
        regime_config: Dict[str, Any]
    ) -> bool:
        """Validate that regime-specific optimization is working correctly.
        
        Args:
            regime_id: Regime ID
            optimization_results: Optimization results
            regime_config: Regime configuration
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            validation_passed = True
            validation_issues = []
            
            # Check if optimization was performed
            if optimization_results.get('status') != 'optimized':
                validation_issues.append(f"Optimization status: {optimization_results.get('status', 'unknown')}")
                validation_passed = False
            
            # Check if regime-specific periods were found
            regime_specific_periods = optimization_results.get('optimization_results', {}).get('regime_specific_periods', {})
            regime_key = f'regime_{regime_id}'
            
            if regime_key not in regime_specific_periods:
                validation_issues.append(f"No regime-specific periods found for {regime_key}")
                # This might not be a failure if global optimization was used
            
            # Check if optimized periods are being used
            optimized_periods = regime_config.get('optimized_periods', {})
            if not optimized_periods:
                validation_issues.append("No optimized periods in regime config")
                validation_passed = False
            
            # Check if interaction patterns were updated
            interaction_patterns = regime_config.get('interaction_patterns', {})
            if not interaction_patterns:
                validation_issues.append("No interaction patterns configured")
                validation_passed = False
            
            # Log validation results
            if validation_passed:
                self.logger.info(f"✅ Regime {regime_id} optimization validation passed")
            else:
                self.logger.warning(f"⚠️ Regime {regime_id} optimization validation issues: {validation_issues}")
            
            return validation_passed
            
        except Exception as e:
            self.logger.error(f"❌ Error validating regime {regime_id} optimization: {e}")
            return False
    
    async def _load_labeled_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load labeled data from previous step.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Labeled DataFrame or None
        """
        try:
            # Try per-regime labeled data first
            labeled_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_labeled_per_regime.parquet'
            
            if not labeled_path.exists():
                # Fall back to standard labeled data
                labeled_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_labeled.parquet'
            
            if labeled_path.exists():
                data = pd.read_parquet(labeled_path)
                self.logger.info(f"✅ Loaded labeled data: {len(data)} rows")
                return data
            else:
                self.logger.error(f"❌ Labeled data not found: {labeled_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading labeled data: {e}")
            return None
    
    def _aggregate_regime_features(
        self,
        regime_results: Dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """Aggregate per-regime feature results intelligently.
        
        Args:
            regime_results: Dictionary of regime results
            
        Returns:
            Aggregated DataFrame with all features
        """
        if not regime_results:
            return pd.DataFrame()
        
        # Get common columns across all regimes
        all_columns = set()
        for df in regime_results.values():
            if df is not None:
                all_columns.update(df.columns)
        
        # Base columns that should always be present
        base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                       'composite_cluster_id', 'feature_regime_id']
        
        # Concatenate all regime results
        dfs = []
        for regime_id, df in regime_results.items():
            if df is not None and not df.empty:
                # Ensure all columns exist (fill missing with NaN)
                for col in all_columns:
                    if col not in df.columns:
                        df[col] = np.nan
                
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # Concatenate and sort by timestamp
        aggregated = pd.concat(dfs, ignore_index=True)
        aggregated = aggregated.sort_values('timestamp').reset_index(drop=True)
        
        return aggregated
    
    async def _save_regime_feature_metadata(
        self,
        regime_feature_info: Dict[int, Dict[str, Any]],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> None:
        """Save metadata about regime-specific features.
        
        Args:
            regime_feature_info: Feature information for each regime
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
        """
        try:
            metadata = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'total_regimes': len(regime_feature_info),
                'regime_features': regime_feature_info,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            metadata_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_regime_features_metadata.json'
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"✅ Saved regime feature metadata: {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving feature metadata: {e}")
    
    def _log_feature_statistics(
        self,
        aggregated_data: pd.DataFrame,
        regime_feature_info: Dict[int, Dict[str, Any]]
    ) -> None:
        """Log statistics about the engineered features.
        
        Args:
            aggregated_data: Aggregated feature data
            regime_feature_info: Feature information per regime
        """
        try:
            self.logger.info("📊 Feature Engineering Statistics:")
            self.logger.info(f"   Total samples: {len(aggregated_data)}")
            
            # Get feature columns
            feature_cols = [c for c in aggregated_data.columns 
                          if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'composite_cluster_id', 'feature_regime_id', 'label']]
            
            self.logger.info(f"   Total features: {len(feature_cols)}")
            
            # Per-regime statistics
            self.logger.info("📊 Per-Regime Feature Statistics:")
            for regime_id, info in regime_feature_info.items():
                if info is None:
                    continue
                    
                self.logger.info(f"   Regime {regime_id}:")
                self.logger.info(f"      Features: {info.get('num_features', 0)}")
                self.logger.info(f"      Samples: {info.get('data_shape', [0])[0]}")
                self.logger.info(f"      Context rows: {info.get('context_rows', 0)}")
                
                # Log regime emphasis and optimization info if available
                config = info.get('regime_config', {})
                optimization_info = info.get('optimization_info', {})
                
                if 'emphasis' in config:
                    self.logger.info(f"      Emphasis: {config['emphasis']}")
                
                if optimization_info:
                    self.logger.info(f"      Optimization Priority: {optimization_info.get('optimization_priority', 'N/A')}")
                    self.logger.info(f"      Adaptive Lookback: {optimization_info.get('adaptive_lookback_enabled', False)}")
                    
                    optimized_periods = optimization_info.get('optimized_periods', {})
                    if optimized_periods:
                        period_count = sum(len(periods.get('selected_periods', [])) for periods in optimized_periods.values())
                        self.logger.info(f"      Optimized Indicators: {len(optimized_periods)} ({period_count} total periods)")
                    else:
                        self.logger.info(f"      Optimized Indicators: None (using fallback)")
                    
        except Exception as e:
            self.logger.error(f"❌ Error logging feature statistics: {e}")


@traced(span_name='run_per_regime_feature_engineering_step')
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
    """Run the enhanced per-regime feature engineering step.
    
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
    logger.info("🚀 Starting Step 6: Per-Regime Feature Engineering")
    
    if config is None:
        config = {}
    
    # Load default per-regime configuration
    try:
        config_path = Path(__file__).parent / 'step06_per_regime_config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                default_config = json.load(f)
                # Merge with user config, user config takes precedence
                config = {**default_config, **config}
                logger.info("✅ Loaded per-regime feature engineering configuration")
        else:
            logger.warning("⚠️ Per-regime config file not found, using defaults")
    except Exception as e:
        logger.warning(f"⚠️ Error loading per-regime config: {e}, using defaults")
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Ensure per-regime processing is enabled
    config['per_regime_feature_engineering'] = True
    
    # Initialize and run the per-regime feature engineering step
    step = PerRegimeFeatureEngineeringStep(config)
    await step.initialize()
    
    success = await step.execute_per_regime_feature_engineering(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )
    
    if success:
        logger.info("✅ Step 6: Per-Regime Feature Engineering completed successfully")
    else:
        logger.error("❌ Step 6: Per-Regime Feature Engineering failed")
        
    return success


if __name__ == '__main__':
    async def test():
        """Test the per-regime feature engineering step."""
        success = await run_per_regime_step(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        print(f'Per-regime feature engineering result: {success}')
        
    asyncio.run(test())