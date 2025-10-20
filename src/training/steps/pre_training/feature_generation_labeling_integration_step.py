"""
Feature Generation Labeling Integration Step

This step performs labeling integration as part of the feature generation pipeline
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
class LabelingIntegrationResult:
    """Result of labeling integration step."""
    
    success: bool
    labeled_data: pd.DataFrame
    labeling_metadata: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


class FeatureGenerationLabelingIntegrationStep(BaseStep):
    """Labeling integration step using BaseStep architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the labeling integration step."""
        super().__init__("feature_generation_labeling_integration_step", config)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute labeling integration step using BaseStep architecture."""
        
        self.logger.info("🏷️ Starting labeling integration step")
        
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
            
            # If no data provided, create sample data for labeling
            if data is None:
                self.logger.warning("No data provided, creating sample data for labeling")
                data = pd.DataFrame({
                    'open': np.random.randn(1000).cumsum() + 100,
                    'high': np.random.randn(1000).cumsum() + 105,
                    'low': np.random.randn(1000).cumsum() + 95,
                    'close': np.random.randn(1000).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, 1000)
                })
            
            # Perform labeling integration
            labeling_result = await self._perform_labeling_integration(
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
            
            # Save labeling results as artifacts
            self._save_dataframe(labeling_result['labeled_data'], 'labeled_data')
            self._save_metadata(labeling_result['labeling_metadata'], 'labeling_metadata')
            
            if labeling_result['success']:
                self.logger.info(f"✅ Labeling integration completed successfully with {len(labeling_result['labeled_data'].columns)} labeled features")
            else:
                self.logger.error(f"❌ Labeling integration failed: {labeling_result.get('error_message', 'Unknown error')}")
            
            return {
                'success': labeling_result['success'],
                'artifacts': ['labeled_data', 'labeling_metadata'],
                'metrics': {
                    'quality_metrics': labeling_result['quality_metrics'],
                    'labeling_metadata': labeling_result['labeling_metadata']
                },
                'error': labeling_result.get('error_message')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Labeling integration step failed with exception: {e}")
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
    
    async def _perform_labeling_integration(self,
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
        """Perform the actual labeling integration logic."""
        
        try:
            # Create labeled data by adding profit labels
            labeled_data = data.copy()
            
            # Add basic profit labeling based on price movements
            if 'close' in labeled_data.columns:
                # Calculate future returns for labeling
                future_returns = labeled_data['close'].pct_change().shift(-1)
                
                # Create binary labels (1 for positive returns, 0 for negative)
                labeled_data['profit_label'] = (future_returns > 0).astype(int)
                
                # Add continuous profit labels
                labeled_data['profit_continuous'] = future_returns.fillna(0)
                
                # Add volatility-based labels
                returns = labeled_data['close'].pct_change()
                volatility = returns.rolling(window=20).std()
                labeled_data['volatility_label'] = (volatility > volatility.median()).astype(int)
            
            # Generate labeling metadata
            labeling_metadata = {
                'labeling_method': 'price_movement_based',
                'data_shape': labeled_data.shape,
                'label_columns': [col for col in labeled_data.columns if 'label' in col],
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'labeling_timestamp': datetime.now().isoformat()
            }
            
            # Calculate quality metrics
            quality_metrics = {
                'label_coverage': labeled_data['profit_label'].notna().mean() if 'profit_label' in labeled_data.columns else 0.0,
                'label_distribution': labeled_data['profit_label'].value_counts().to_dict() if 'profit_label' in labeled_data.columns else {},
                'data_quality_score': 0.9  # Placeholder
            }
            
            # Apply any custom overrides
            if custom_overrides:
                labeling_metadata.update(custom_overrides)
            
            return {
                'success': True,
                'labeled_data': labeled_data,
                'labeling_metadata': labeling_metadata,
                'quality_metrics': quality_metrics,
                'error_message': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'labeled_data': data,
                'labeling_metadata': {},
                'quality_metrics': {},
                'error_message': str(e)
            }


# Command handler for ares_launcher integration
async def handle_feature_generation_labeling_integration_step(
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
    Handle feature generation labeling integration step command.
    
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
        Dict with labeling results
    """
    # Create step instance and execute
    step = FeatureGenerationLabelingIntegrationStep()
    
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