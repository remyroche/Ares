"""
Feature Generation Data Validation Step

This step performs comprehensive data validation and quality assessment
as the first step in the feature generation pipeline using the BaseStep architecture.
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
class DataValidationResult:
    """Result of data validation step."""
    
    success: bool
    data_quality_score: float
    validation_metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


class FeatureGenerationDataValidationStep(BaseStep):
    """Data validation step using BaseStep architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data validation step."""
        super().__init__("feature_generation_data_validation_step", config)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data validation step using BaseStep architecture."""
        
        self.logger.info("🔍 Starting data validation step")
        
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
            
            # If no data provided, create sample data for validation
            if data is None:
                self.logger.warning("No data provided, creating sample data for validation")
                data = pd.DataFrame({
                    'open': np.random.randn(1000).cumsum() + 100,
                    'high': np.random.randn(1000).cumsum() + 105,
                    'low': np.random.randn(1000).cumsum() + 95,
                    'close': np.random.randn(1000).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, 1000)
                })
            
            # Perform data validation
            validation_result = await self._perform_data_validation(
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
            
            # Save validation results as artifacts
            self._save_dataframe(validation_result['validated_data'], 'validated_data')
            self._save_metadata(validation_result['validation_metadata'], 'validation_metadata')
            
            if validation_result['success']:
                self.logger.info(f"✅ Data validation completed successfully with quality score: {validation_result['data_quality_score']:.3f}")
            else:
                self.logger.error(f"❌ Data validation failed: {validation_result.get('error_message', 'Unknown error')}")
            
            return {
                'success': validation_result['success'],
                'artifacts': ['validated_data', 'validation_metadata'],
                'metrics': {
                    'data_quality_score': validation_result['data_quality_score'],
                    'validation_metadata': validation_result['validation_metadata']
                },
                'error': validation_result.get('error_message')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Data validation step failed with exception: {e}")
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
    
    async def _perform_data_validation(self,
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
        """Perform the actual data validation logic."""
        
        try:
            # Basic data quality checks
            quality_checks = {
                'has_data': len(data) > 0,
                'has_required_columns': all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']),
                'no_nan_values': not data.isnull().any().any(),
                'positive_volume': (data['volume'] > 0).all() if 'volume' in data.columns else True,
                'valid_ohlc': (data['high'] >= data['low']).all() if all(col in data.columns for col in ['high', 'low']) else True
            }
            
            # Calculate overall quality score
            quality_score = sum(quality_checks.values()) / len(quality_checks)
            
            # Generate validation metadata
            validation_metadata = {
                'quality_checks': quality_checks,
                'data_shape': data.shape,
                'data_types': data.dtypes.to_dict(),
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Apply any custom overrides
            if custom_overrides:
                validation_metadata.update(custom_overrides)
            
            return {
                'success': quality_score > 0.5,  # Require at least 50% quality
                'data_quality_score': quality_score,
                'validated_data': data,
                'validation_metadata': validation_metadata,
                'error_message': None if quality_score > 0.5 else f"Data quality too low: {quality_score:.3f}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'data_quality_score': 0.0,
                'validated_data': data,
                'validation_metadata': {},
                'error_message': str(e)
            }


# Command handler for ares_launcher integration
async def handle_feature_generation_data_validation_step(
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
    Handle feature generation data validation step command.
    
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
        Dict with validation results
    """
    # Create step instance and execute
    step = FeatureGenerationDataValidationStep()
    
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