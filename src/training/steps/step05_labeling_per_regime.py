"""Enhanced Step 5: Per-Regime Labeling.

This module provides per-HMM regime labeling functionality, ensuring that
labeling is performed on a per-regime basis for better regime-specific modeling.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from src.training.steps.step05_labeling import LabelingStep
from src.training.steps.regime_handler import regime_handler
from src.training.steps.regime_processing_decorator import (
    per_regime_processing, 
    aggregate_regime_results,
    RegimeProcessingContext
)
from src.utils.logger import getChild as get_logger
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import traced, validates, handles_errors


logger = get_logger('Step5LabelingPerRegime')


class PerRegimeLabelingStep(LabelingStep):
    """Enhanced labeling step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_labeling', True)
        self.regime_specific_params = config.get('regime_specific_params', {})
        
    @traced(span_name='execute_per_regime_labeling')
    async def execute_per_regime_labeling(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool = False
    ) -> bool:
        """Execute labeling on a per-regime basis.
        
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
            self.logger.info("🚀 Starting per-regime labeling process")
            
            # Use regime processing context
            async with RegimeProcessingContext(symbol, exchange, timeframe, data_dir) as ctx:
                if ctx.regime_data is None:
                    self.logger.error("❌ Failed to load regime data")
                    return False
                    
                self.logger.info(f"📊 Processing {len(ctx.regime_ids)} regimes separately")
                
                # Process each regime
                regime_results = {}
                for regime_id in ctx.regime_ids:
                    self.logger.info(f"🔄 Processing regime {regime_id}")
                    
                    # Get regime-specific parameters if available
                    regime_params = self.regime_specific_params.get(
                        f'regime_{regime_id}', 
                        self.get_default_regime_params(regime_id)
                    )
                    
                    # Process this regime
                    result = await self._label_single_regime(
                        ctx=ctx,
                        regime_id=regime_id,
                        regime_params=regime_params
                    )
                    
                    if result is not None:
                        regime_results[regime_id] = result
                        self.logger.info(f"✅ Successfully labeled regime {regime_id}")
                    else:
                        self.logger.error(f"❌ Failed to label regime {regime_id}")
                
                # Save per-regime results
                success = await regime_handler.save_regime_results(
                    results=regime_results,
                    step_name='step05_labeling',
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data_dir=data_dir,
                    result_type='labeled_data'
                )
                
                if success:
                    # Aggregate results for unified output
                    aggregated = aggregate_regime_results(regime_results, 'concat')
                    
                    # Save aggregated results
                    output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_labeled_per_regime.parquet'
                    aggregated.to_parquet(output_path, index=False)
                    self.logger.info(f"✅ Saved aggregated labeled data: {output_path}")
                    
                    # Calculate and log statistics
                    self._log_labeling_statistics(aggregated, regime_results)
                    
                return success
                
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime labeling: {e}")
            return False
    
    async def _label_single_regime(
        self,
        ctx: RegimeProcessingContext,
        regime_id: int,
        regime_params: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Label data for a single regime.
        
        Args:
            ctx: Regime processing context
            regime_id: Regime ID to process
            regime_params: Parameters for this regime
            
        Returns:
            Labeled DataFrame or None
        """
        try:
            # Get regime data with context
            regime_data = ctx.get_regime_data(regime_id, preserve_context=True)
            
            if regime_data.empty:
                self.logger.warning(f"⚠️ No data for regime {regime_id}")
                return None
            
            # Update labeling parameters for this regime
            self.time_barrier_minutes = regime_params.get('time_barrier_minutes', 30)
            self.max_lookahead = regime_params.get('max_lookahead', 100)
            self.profit_take_multiplier = regime_params.get('profit_take_multiplier', 0.002)
            self.stop_loss_multiplier = regime_params.get('stop_loss_multiplier', 0.001)
            
            # Remove context indicator before labeling
            context_mask = regime_data.get('is_regime_context', pd.Series(False, index=regime_data.index))
            if 'is_regime_context' in regime_data.columns:
                regime_data = regime_data.drop(columns=['is_regime_context'])
            
            # Apply labeling
            labeled_data = await self._apply_labeling_method(regime_data)
            
            if labeled_data is None:
                return None
            
            # Add regime ID to labeled data
            labeled_data['labeled_regime_id'] = regime_id
            
            # Mark context rows (if any) with special label
            if context_mask.any():
                labeled_data.loc[context_mask, 'label'] = -999  # Special value for context
                labeled_data.loc[context_mask, 'label_type'] = 'context'
            
            return labeled_data
            
        except Exception as e:
            self.logger.error(f"❌ Error labeling regime {regime_id}: {e}")
            return None
    
    def get_default_regime_params(self, regime_id: int) -> Dict[str, Any]:
        """Get default parameters for a regime based on its ID.
        
        Different regimes may benefit from different labeling parameters.
        This method provides sensible defaults based on regime characteristics.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific parameters
        """
        # Default base parameters
        base_params = {
            'time_barrier_minutes': 30,
            'max_lookahead': 100,
            'profit_take_multiplier': 0.002,
            'stop_loss_multiplier': 0.001
        }
        
        # Adjust based on regime ID patterns
        # Lower regime IDs often correspond to calmer market conditions
        # Higher regime IDs often correspond to more volatile conditions
        
        if regime_id <= 2:
            # Calm/trending regimes - wider barriers, longer time
            return {
                **base_params,
                'time_barrier_minutes': 45,
                'max_lookahead': 150,
                'profit_take_multiplier': 0.003,
                'stop_loss_multiplier': 0.0015
            }
        elif regime_id >= 5:
            # Volatile regimes - tighter barriers, shorter time
            return {
                **base_params,
                'time_barrier_minutes': 20,
                'max_lookahead': 75,
                'profit_take_multiplier': 0.0015,
                'stop_loss_multiplier': 0.0008
            }
        else:
            # Medium regimes - standard parameters
            return base_params
    
    def _log_labeling_statistics(
        self, 
        aggregated_data: pd.DataFrame,
        regime_results: Dict[int, pd.DataFrame]
    ) -> None:
        """Log statistics about the labeling results.
        
        Args:
            aggregated_data: Aggregated labeled data
            regime_results: Per-regime results
        """
        try:
            total_samples = len(aggregated_data)
            
            # Filter out context rows
            valid_data = aggregated_data[aggregated_data.get('label', 0) != -999]
            valid_samples = len(valid_data)
            
            self.logger.info("📊 Labeling Statistics:")
            self.logger.info(f"   Total samples: {total_samples}")
            self.logger.info(f"   Valid samples: {valid_samples}")
            self.logger.info(f"   Context samples: {total_samples - valid_samples}")
            
            # Per-regime statistics
            self.logger.info("📊 Per-Regime Statistics:")
            for regime_id, regime_data in regime_results.items():
                if regime_data is None:
                    continue
                    
                regime_valid = regime_data[regime_data.get('label', 0) != -999]
                regime_total = len(regime_data)
                regime_valid_count = len(regime_valid)
                
                if 'label' in regime_valid.columns and regime_valid_count > 0:
                    label_dist = regime_valid['label'].value_counts()
                    
                    self.logger.info(f"   Regime {regime_id}:")
                    self.logger.info(f"      Total: {regime_total}")
                    self.logger.info(f"      Valid: {regime_valid_count}")
                    
                    for label, count in label_dist.items():
                        pct = (count / regime_valid_count) * 100
                        self.logger.info(f"      Label {label}: {count} ({pct:.1f}%)")
                        
        except Exception as e:
            self.logger.error(f"❌ Error logging statistics: {e}")


@traced(span_name='run_per_regime_labeling_step')
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
    """Run the enhanced per-regime labeling step.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun the step
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("🚀 Starting Step 5: Per-Regime Labeling")
    
    if config is None:
        config = {}
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Enable per-regime processing
    config['per_regime_labeling'] = True
    
    # Initialize and run the per-regime labeling step
    step = PerRegimeLabelingStep(config)
    await step.initialize()
    
    success = await step.execute_per_regime_labeling(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )
    
    if success:
        logger.info("✅ Step 5: Per-Regime Labeling completed successfully")
    else:
        logger.error("❌ Step 5: Per-Regime Labeling failed")
        
    return success


# Decorator-based approach for simpler usage
@per_regime_processing(result_type='labeled_data', parallel=True)
async def process_labeling_regime(
    data: pd.DataFrame,
    regime_id: int,
    **kwargs
) -> pd.DataFrame:
    """Process labeling for a single regime using decorator approach.
    
    Args:
        data: Regime data
        regime_id: Regime ID
        **kwargs: Additional arguments
        
    Returns:
        Labeled DataFrame
    """
    # Create a temporary labeling step with regime-specific config
    config = kwargs.get('config', {})
    step = LabelingStep(config)
    
    # Apply labeling
    labeled_data = await step._apply_labeling_method(data)
    
    if labeled_data is not None:
        labeled_data['labeled_regime_id'] = regime_id
        
    return labeled_data


if __name__ == '__main__':
    async def test():
        """Test the per-regime labeling step."""
        success = await run_per_regime_step(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        print(f'Per-regime labeling result: {success}')
        
    asyncio.run(test())