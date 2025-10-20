"""
Feature Generation Feature Generation Step

This step generates features as part of the feature generation pipeline
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
class FeatureGenerationResult:
    """Result of feature generation step."""
    
    success: bool
    generated_features: pd.DataFrame
    feature_metadata: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


class FeatureGenerationFeatureGenerationStep(BaseStep):
    """Feature generation step using BaseStep architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature generation step."""
        super().__init__("feature_generation_feature_generation_step", config)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature generation step using BaseStep architecture."""
        
        self.logger.info("🔧 Starting feature generation step")
        
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
            
            # If no data provided, create sample data for feature generation
            if data is None:
                self.logger.warning("No data provided, creating sample data for feature generation")
                data = pd.DataFrame({
                    'open': np.random.randn(1000).cumsum() + 100,
                    'high': np.random.randn(1000).cumsum() + 105,
                    'low': np.random.randn(1000).cumsum() + 95,
                    'close': np.random.randn(1000).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, 1000)
                })
            
            # Perform feature generation
            generation_result = await self._perform_feature_generation(
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
            
            # Save generated features as artifacts
            self._save_dataframe(generation_result['generated_features'], 'generated_features')
            self._save_metadata(generation_result['feature_metadata'], 'feature_metadata')
            
            if generation_result['success']:
                self.logger.info(f"✅ Feature generation completed successfully with {len(generation_result['generated_features'].columns)} features")
            else:
                self.logger.error(f"❌ Feature generation failed: {generation_result.get('error_message', 'Unknown error')}")
            
            return {
                'success': generation_result['success'],
                'artifacts': ['generated_features', 'feature_metadata'],
                'metrics': {
                    'generation_metrics': generation_result['generation_metrics'],
                    'feature_metadata': generation_result['feature_metadata']
                },
                'error': generation_result.get('error_message')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Feature generation step failed with exception: {e}")
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
    
    async def _perform_feature_generation(self,
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
        """Perform the actual feature generation logic."""
        
        try:
            # Create generated features by adding technical indicators
            generated_features = data.copy()
            
            # Add technical indicators as features
            if 'close' in generated_features.columns:
                # Moving averages
                generated_features['sma_5'] = generated_features['close'].rolling(window=5).mean()
                generated_features['sma_20'] = generated_features['close'].rolling(window=20).mean()
                generated_features['sma_50'] = generated_features['close'].rolling(window=50).mean()
                
                # Exponential moving averages
                generated_features['ema_12'] = generated_features['close'].ewm(span=12).mean()
                generated_features['ema_26'] = generated_features['close'].ewm(span=26).mean()
                
                # RSI
                delta = generated_features['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                generated_features['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                bb_period = 20
                bb_std = 2
                bb_middle = generated_features['close'].rolling(window=bb_period).mean()
                bb_std_val = generated_features['close'].rolling(window=bb_period).std()
                generated_features['bb_upper'] = bb_middle + (bb_std_val * bb_std)
                generated_features['bb_lower'] = bb_middle - (bb_std_val * bb_std)
                generated_features['bb_width'] = generated_features['bb_upper'] - generated_features['bb_lower']
                
                # MACD
                ema_12 = generated_features['close'].ewm(span=12).mean()
                ema_26 = generated_features['close'].ewm(span=26).mean()
                generated_features['macd'] = ema_12 - ema_26
                generated_features['macd_signal'] = generated_features['macd'].ewm(span=9).mean()
                generated_features['macd_histogram'] = generated_features['macd'] - generated_features['macd_signal']
                
                # Price-based features
                generated_features['price_change'] = generated_features['close'].pct_change()
                generated_features['price_range'] = (generated_features['high'] - generated_features['low']) / generated_features['close']
                generated_features['body_size'] = abs(generated_features['close'] - generated_features['open']) / generated_features['close']
                
                # Volume features
                if 'volume' in generated_features.columns:
                    generated_features['volume_sma'] = generated_features['volume'].rolling(window=20).mean()
                    generated_features['volume_ratio'] = generated_features['volume'] / generated_features['volume_sma']
                    generated_features['price_volume'] = generated_features['close'] * generated_features['volume']
            
            # Generate feature metadata
            feature_metadata = {
                'feature_generation_method': 'technical_indicators',
                'original_features': list(data.columns),
                'generated_features': [col for col in generated_features.columns if col not in data.columns],
                'total_features': len(generated_features.columns),
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
                'features_generated': len(generated_features.columns) - len(data.columns),
                'data_shape': generated_features.shape,
                'feature_types': {
                    'technical_indicators': len([col for col in generated_features.columns if any(indicator in col for indicator in ['sma', 'ema', 'rsi', 'bb', 'macd'])]),
                    'price_features': len([col for col in generated_features.columns if any(price in col for price in ['price_change', 'price_range', 'body_size'])]),
                    'volume_features': len([col for col in generated_features.columns if 'volume' in col])
                },
                'missing_values': generated_features.isnull().sum().sum()
            }
            
            # Apply any custom overrides
            if custom_overrides:
                feature_metadata.update(custom_overrides)
            
            return {
                'success': True,
                'generated_features': generated_features,
                'feature_metadata': feature_metadata,
                'generation_metrics': generation_metrics,
                'error_message': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'generated_features': data,
                'feature_metadata': {},
                'generation_metrics': {},
                'error_message': str(e)
            }


# Command handler for ares_launcher integration
async def handle_feature_generation_feature_generation_step(
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
    Handle feature generation feature generation step command.
    
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
        Dict with feature generation results
    """
    # Create step instance and execute
    step = FeatureGenerationFeatureGenerationStep()
    
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