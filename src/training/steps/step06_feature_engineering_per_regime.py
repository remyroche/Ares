"""Enhanced Step 6: Per-Regime Feature Engineering.

This module provides per-HMM regime feature engineering functionality, ensuring that
features are engineered specifically for each regime's characteristics.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np

from src.training.steps.step06_feature_engineering import FeatureEngineeringStep
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


class PerRegimeFeatureEngineeringStep(FeatureEngineeringStep):
    """Enhanced feature engineering step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_feature_engineering', True)
        self.regime_specific_features = config.get('regime_specific_features', {})
        self.adaptive_lookback = config.get('adaptive_lookback_per_regime', True)
        
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
                    optimization_results = await self.optimize_lookback_periods(
                        optimization_data,
                        target,
                        regimes=pd.Series(regime_id, index=optimization_data.index)
                    )
                    
                    # Update feature configuration with optimized lookbacks
                    if optimization_results.get('selected_features'):
                        regime_config['optimized_features'] = optimization_results['selected_features']
                        self.logger.info(f"✅ Found {len(optimization_results['selected_features'])} optimized features")
            
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
                'context_rows': int(context_mask.sum()) if context_mask is not None else 0
            }
            
            return features_df, feature_info
            
        except Exception as e:
            self.logger.error(f"❌ Error engineering features for regime {regime_id}: {e}")
            return None, None
    
    def _get_regime_feature_config(self, regime_id: int) -> Dict[str, Any]:
        """Get feature engineering configuration for a specific regime.
        
        Different regimes may benefit from different feature sets and parameters.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific feature configuration
        """
        # Check if custom config exists for this regime
        if f'regime_{regime_id}' in self.regime_specific_features:
            return self.regime_specific_features[f'regime_{regime_id}']
        
        # Otherwise, create adaptive configuration based on regime characteristics
        base_config = {
            'enable_technical_indicators': True,
            'enable_price_features': True,
            'enable_volume_features': True,
            'enable_volatility_features': True,
            'enable_microstructure_features': True
        }
        
        # Adapt based on regime ID patterns
        if regime_id <= 2:
            # Low regime IDs - often trending markets
            # Emphasize trend-following indicators
            return {
                **base_config,
                'lookback_periods': [10, 20, 50, 100, 200],
                'emphasis': 'trend',
                'additional_features': [
                    'SMA_cross_features',
                    'EMA_ribbon',
                    'ADX_features',
                    'trend_strength'
                ]
            }
        elif regime_id >= 5:
            # High regime IDs - often volatile/ranging markets
            # Emphasize mean-reversion and volatility indicators
            return {
                **base_config,
                'lookback_periods': [5, 10, 20, 30],
                'emphasis': 'mean_reversion',
                'additional_features': [
                    'RSI_divergence',
                    'Bollinger_bands_features',
                    'ATR_bands',
                    'volatility_cones'
                ]
            }
        else:
            # Medium regime IDs - balanced approach
            return {
                **base_config,
                'lookback_periods': [7, 14, 30, 60],
                'emphasis': 'balanced',
                'additional_features': [
                    'momentum_features',
                    'volume_profile',
                    'market_microstructure'
                ]
            }
    
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
                
                # Log regime emphasis if available
                config = info.get('regime_config', {})
                if 'emphasis' in config:
                    self.logger.info(f"      Emphasis: {config['emphasis']}")
                    
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
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Enable per-regime processing
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