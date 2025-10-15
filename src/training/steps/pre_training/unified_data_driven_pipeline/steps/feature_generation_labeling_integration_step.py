"""
Feature Generation Labeling Integration Step

This step performs labeling integration as part of the unified data-driven pipeline
by calling the consolidated pipeline at the appropriate stage.
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
    run_labeling_integration_step
)


@dataclass
class LabelingIntegrationResult:
    """Result of labeling integration step."""
    
    success: bool
    labeled_data: pd.DataFrame
    labeling_metadata: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


class FeatureGenerationLabelingIntegrationStep:
    """Labeling integration step that calls the consolidated pipeline."""
    
    def __init__(self):
        """Initialize the labeling integration step."""
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
                     custom_overrides: Optional[Dict[str, Any]] = None) -> LabelingIntegrationResult:
        """Execute labeling integration step using consolidated pipeline."""
        
        self.logger.info("🏷️ Starting labeling integration step using consolidated pipeline")
        
        try:
            # Call the consolidated pipeline runner
            result = await run_labeling_integration_step(
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
            
            # Convert result to LabelingIntegrationResult
            labeling_result = LabelingIntegrationResult(
                success=result['success'],
                labeled_data=result.get('labeled_data', pd.DataFrame()),
                labeling_metadata=result.get('labeling_metadata', {}),
                quality_metrics=result.get('quality_metrics', {}),
                artifacts=result.get('artifacts', {}),
                error_message=result.get('error_message')
            )
            
            if labeling_result.success:
                self.logger.info(f"✅ Labeling integration completed successfully with {len(labeling_result.labeled_data.columns)} labeled features")
            else:
                self.logger.error(f"❌ Labeling integration failed: {labeling_result.error_message}")
            
            return labeling_result
            
        except Exception as e:
            self.logger.error(f"❌ Labeling integration step failed with exception: {e}")
            return LabelingIntegrationResult(
                success=False,
                labeled_data=pd.DataFrame(),
                labeling_metadata={},
                quality_metrics={},
                artifacts={},
                error_message=str(e)
            )


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
) -> LabelingIntegrationResult:
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
        LabelingIntegrationResult with labeling results
    """
    # Create sample data for labeling integration (in real usage, this would come from data loading)
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Create step instance and execute
    step = FeatureGenerationLabelingIntegrationStep()
    
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