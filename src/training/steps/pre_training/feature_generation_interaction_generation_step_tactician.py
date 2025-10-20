"""
Feature Generation Interaction Generation Step - Tactician

This step generates feature interactions for the Tactician model as part of the feature generation pipeline
using the BaseStep architecture.
"""

from __future__ import annotations

import logging
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.base_step import BaseStep


@dataclass
class InteractionGenerationResult:
    """Result of interaction generation step."""
    
    success: bool
    interaction_features: pd.DataFrame
    interaction_metadata: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


class FeatureGenerationInteractionGenerationStepTactician(BaseStep):
    """Interaction generation step for Tactician model using BaseStep architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the interaction generation step for Tactician."""
        super().__init__("feature_generation_interaction_generation_step_tactician", config)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute interaction generation step for Tactician using BaseStep architecture."""
        
        self.logger.info("🔗 Starting interaction generation step for Tactician model")
        
        try:
            # Extract parameters from config
            data = config.get('data')
            symbol = config.get('symbol', 'ETHUSDT')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'longs')
            intensity = config.get('intensity', 'blank')
            lookback_days = config.get('lookback_days')
            start_date = config.get('start_date')
            end_date = config.get('end_date')
            exchange = config.get('exchange', 'binance')
            custom_overrides = config.get('custom_overrides')
            
            # If no data provided, create sample data for interaction generation
            if data is None:
                self.logger.warning("No data provided, creating sample data for interaction generation")
                data = pd.DataFrame({
                    'open': np.random.randn(1000).cumsum() + 100,
                    'high': np.random.randn(1000).cumsum() + 105,
                    'low': np.random.randn(1000).cumsum() + 95,
                    'close': np.random.randn(1000).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, 1000)
                })
                # Add some technical indicators
                data['sma_20'] = data['close'].rolling(window=20).mean()
                data['rsi'] = 50 + np.random.randn(1000) * 10
                data['macd'] = np.random.randn(1000)
            
            # Perform interaction generation for Tactician
            generation_result = await self._perform_interaction_generation_tactician(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                intensity=intensity,
                lookback_days=lookback_days,
                start_date=start_date,
                end_date=end_date,
                exchange=exchange,
                custom_overrides=custom_overrides
            )
            
            # Save interaction results as artifacts
            self._save_dataframe(generation_result['interaction_features'], 'interaction_features_tactician')
            self._save_metadata(generation_result['interaction_metadata'], 'interaction_metadata_tactician')
            
            if generation_result['success']:
                self.logger.info(f"✅ Interaction generation for Tactician completed successfully with {len(generation_result['interaction_features'].columns)} interaction features")
            else:
                self.logger.error(f"❌ Interaction generation for Tactician failed: {generation_result.get('error_message', 'Unknown error')}")
            
            return {
                'success': generation_result['success'],
                'artifacts': ['interaction_features_tactician', 'interaction_metadata_tactician'],
                'metrics': {
                    'generation_metrics': generation_result['generation_metrics'],
                    'interaction_metadata': generation_result['interaction_metadata']
                },
                'error': generation_result.get('error_message')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Interaction generation step for Tactician failed with exception: {e}")
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
    
    async def _perform_interaction_generation_tactician(self,
                                                      data: pd.DataFrame,
                                                      symbol: str,
                                                      timeframe: str,
                                                      direction: str,
                                                      intensity: str,
                                                      lookback_days: Optional[int],
                                                      start_date: Optional[str],
                                                      end_date: Optional[str],
                                                      exchange: str,
                                                      custom_overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform the actual interaction generation logic for Tactician model."""
        
        try:
            # Get numeric columns for interaction generation
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target columns if they exist
            target_columns = ['profit_label', 'profit_continuous', 'volatility_label']
            feature_columns = [col for col in numeric_columns if col not in target_columns]
            
            if len(feature_columns) < 2:
                return {
                    'success': False,
                    'interaction_features': data,
                    'interaction_metadata': {},
                    'generation_metrics': {},
                    'error_message': 'Insufficient features for interaction generation'
                }
            
            # Generate interactions specific to Tactician model
            interaction_features = data.copy()
            generated_interactions = []
            
            # 1. Momentum-based interactions (Tactician focuses on momentum and trends)
            momentum_columns = [col for col in feature_columns if any(mom_term in col.lower() for mom_term in ['rsi', 'macd', 'momentum', 'change'])]
            if len(momentum_columns) >= 2:
                for i, col1 in enumerate(momentum_columns):
                    for col2 in momentum_columns[i+1:]:
                        interaction_name = f"{col1}_x_{col2}_tactician"
                        interaction_features[interaction_name] = interaction_features[col1] * interaction_features[col2]
                        generated_interactions.append(interaction_name)
            
            # 2. Volatility-based interactions (Tactician model specific)
            volatility_columns = [col for col in feature_columns if any(vol_term in col.lower() for vol_term in ['bb', 'volatility', 'std', 'range'])]
            if len(volatility_columns) >= 2:
                for i, col1 in enumerate(volatility_columns):
                    for col2 in volatility_columns[i+1:]:
                        interaction_name = f"{col1}_x_{col2}_tactician"
                        interaction_features[interaction_name] = interaction_features[col1] * interaction_features[col2]
                        generated_interactions.append(interaction_name)
            
            # 3. Volume-momentum interactions (Tactician model specific)
            if 'volume' in feature_columns:
                for col in momentum_columns:
                    interaction_name = f"volume_x_{col}_tactician"
                    interaction_features[interaction_name] = interaction_features['volume'] * interaction_features[col]
                    generated_interactions.append(interaction_name)
            
            # 4. Cross-timeframe interactions (Tactician model specific)
            timeframe_columns = [col for col in feature_columns if any(tf_term in col.lower() for tf_term in ['sma', 'ema', 'ma'])]
            if len(timeframe_columns) >= 2:
                for i, col1 in enumerate(timeframe_columns):
                    for col2 in timeframe_columns[i+1:]:
                        interaction_name = f"{col1}_x_{col2}_tactician"
                        interaction_features[interaction_name] = interaction_features[col1] * interaction_features[col2]
                        generated_interactions.append(interaction_name)
            
            # 5. Signal-based interactions (Tactician model specific)
            signal_columns = [col for col in feature_columns if any(sig_term in col.lower() for sig_term in ['signal', 'macd', 'rsi'])]
            if len(signal_columns) >= 2:
                for i, col1 in enumerate(signal_columns):
                    for col2 in signal_columns[i+1:]:
                        interaction_name = f"{col1}_x_{col2}_tactician"
                        interaction_features[interaction_name] = interaction_features[col1] * interaction_features[col2]
                        generated_interactions.append(interaction_name)
            
            # 6. Trend-following interactions (Tactician model specific)
            trend_columns = [col for col in feature_columns if any(trend_term in col.lower() for trend_term in ['trend', 'slope', 'direction'])]
            if len(trend_columns) >= 2:
                for i, col1 in enumerate(trend_columns):
                    for col2 in trend_columns[i+1:]:
                        interaction_name = f"{col1}_x_{col2}_tactician"
                        interaction_features[interaction_name] = interaction_features[col1] * interaction_features[col2]
                        generated_interactions.append(interaction_name)
            
            # 7. Difference-based interactions (Tactician model specific)
            if 'close' in feature_columns:
                for col in feature_columns:
                    if col != 'close':
                        interaction_name = f"{col}_minus_close_tactician"
                        interaction_features[interaction_name] = interaction_features[col] - interaction_features['close']
                        generated_interactions.append(interaction_name)
            
            # 8. Exponential interactions (Tactician model specific)
            for col in feature_columns[:3]:  # Limit to first 3 features
                interaction_name = f"{col}_exp_tactician"
                interaction_features[interaction_name] = np.exp(interaction_features[col] / 100)  # Scale down for stability
                generated_interactions.append(interaction_name)
            
            # Generate interaction metadata
            interaction_metadata = {
                'model_type': 'Tactician',
                'interaction_generation_method': 'tactician_specific',
                'original_features': feature_columns,
                'generated_interactions': generated_interactions,
                'total_interactions': len(generated_interactions),
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'generation_timestamp': datetime.now().isoformat()
            }
            
            # Calculate generation metrics
            generation_metrics = {
                'interactions_generated': len(generated_interactions),
                'interaction_types': {
                    'momentum_based': len([i for i in generated_interactions if any(mom in i for mom in ['rsi', 'macd', 'momentum', 'change'])]),
                    'volatility_based': len([i for i in generated_interactions if any(vol in i for vol in ['bb', 'volatility', 'std', 'range'])]),
                    'volume_based': len([i for i in generated_interactions if 'volume' in i]),
                    'timeframe_based': len([i for i in generated_interactions if any(tf in i for tf in ['sma', 'ema', 'ma'])]),
                    'signal_based': len([i for i in generated_interactions if any(sig in i for sig in ['signal', 'macd', 'rsi'])]),
                    'trend_based': len([i for i in generated_interactions if any(trend in i for trend in ['trend', 'slope', 'direction'])]),
                    'difference_based': len([i for i in generated_interactions if 'minus' in i]),
                    'exponential_based': len([i for i in generated_interactions if 'exp' in i])
                },
                'data_shape': interaction_features.shape,
                'missing_values': interaction_features.isnull().sum().sum()
            }
            
            # Apply any custom overrides
            if custom_overrides:
                interaction_metadata.update(custom_overrides)
            
            return {
                'success': True,
                'interaction_features': interaction_features,
                'interaction_metadata': interaction_metadata,
                'generation_metrics': generation_metrics,
                'error_message': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'interaction_features': data,
                'interaction_metadata': {},
                'generation_metrics': {},
                'error_message': str(e)
            }


# Command handler for ares_launcher integration
async def handle_feature_generation_interaction_generation_step_tactician(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Handle feature generation interaction generation step command for Tactician model.
    
    Args:
        symbol: Trading symbol (default: "ETHUSDT")
        timeframe: Timeframe (default: "15m")
        direction: Direction (default: "longs")
        intensity: Pipeline intensity (default: "blank")
        lookback_days: Lookback days (optional)
        start_date: Start date (optional)
        end_date: End date (optional)
        exchange: Exchange (default: "binance")
        custom_overrides: Custom configuration overrides (optional)
        **kwargs: Additional arguments
        
    Returns:
        Dict with interaction generation results for Tactician
    """
    # Create step instance and execute
    step = FeatureGenerationInteractionGenerationStepTactician()
    
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'direction': direction,
        'intensity': intensity,
        'lookback_days': lookback_days,
        'start_date': start_date,
        'end_date': end_date,
        'exchange': exchange,
        'custom_overrides': custom_overrides
    }
    
    return await step.run(config)