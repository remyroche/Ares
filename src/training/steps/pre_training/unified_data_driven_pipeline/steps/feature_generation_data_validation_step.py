"""
Feature Generation Data Validation Step

This step performs comprehensive data validation and quality assessment
as the first step in the unified data-driven pipeline by calling the
consolidated pipeline at the appropriate stage.
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

from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_data_validation_step
)


@dataclass
class DataValidationResult:
    """Result of data validation step."""
    
    success: bool
    data_quality_score: float
    validation_metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


class FeatureGenerationDataValidationStep:
    """Data validation step that calls the consolidated pipeline."""
    
    def __init__(self):
        """Initialize the data validation step."""
        self.logger = logging.getLogger(__name__)
    
    async def execute(self,
                     data: pd.DataFrame,
                     symbol: str = "ETHUSDT",
                     timeframe: str = "15m",
                     direction: str = "longs",
                     intensity: str = "blank",
                     lookback_days: Optional[int] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     exchange: str = "binance",
                     custom_overrides: Optional[Dict[str, Any]] = None) -> DataValidationResult:
        """Execute data validation step using consolidated pipeline."""
        
        self.logger.info("🔍 Starting data validation step using consolidated pipeline")
        
        try:
            # Call the consolidated pipeline runner
            result = await run_data_validation_step(
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
            
            # Convert result to DataValidationResult
            validation_result = DataValidationResult(
                success=result['success'],
                data_quality_score=result.get('data_quality_score', 0.0),
                validation_metadata=result.get('validation_metadata', {}),
                artifacts=result.get('artifacts', {}),
                error_message=result.get('error_message')
            )
            
            if validation_result.success:
                self.logger.info(f"✅ Data validation completed successfully with quality score: {validation_result.data_quality_score:.3f}")
            else:
                self.logger.error(f"❌ Data validation failed: {validation_result.error_message}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Data validation step failed with exception: {e}")
            return DataValidationResult(
                success=False,
                data_quality_score=0.0,
                validation_metadata={},
                artifacts={},
                error_message=str(e)
            )


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
) -> DataValidationResult:
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
        DataValidationResult with validation results
    """
    # Create sample data for validation (in real usage, this would come from data loading)
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Create step instance and execute
    step = FeatureGenerationDataValidationStep()
    
    return await step.execute(
        data=sample_data,
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