"""
Feature Generation Feature Selection Step

This step performs feature selection as part of the unified data-driven pipeline
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
    run_feature_selection_step
)


@dataclass
class FeatureSelectionResult:
    """Result of feature selection step."""
    
    success: bool
    selected_features: pd.DataFrame
    selection_metadata: Dict[str, Any]
    selection_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


class FeatureGenerationFeatureSelectionStep:
    """Feature selection step that calls the consolidated pipeline."""
    
    def __init__(self):
        """Initialize the feature selection step."""
        self.logger = logging.getLogger(__name__)
    
    async def execute(self,
                     data: pd.DataFrame,
                     targets: pd.Series,
                     symbol: str = "ETHUSDT",
                     timeframe: str = "15m",
                     direction: str = "longs",
                     intensity: str = "blank",
                     lookback_days: Optional[int] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     exchange: str = "binance",
                     custom_overrides: Optional[Dict[str, Any]] = None) -> FeatureSelectionResult:
        """Execute feature selection step using consolidated pipeline."""
        
        self.logger.info("🎯 Starting feature selection step using consolidated pipeline")
        
        try:
            # Call the consolidated pipeline runner
            result = await run_feature_selection_step(
                data=data,
                targets=targets,
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
            
            # Convert result to FeatureSelectionResult
            selection_result = FeatureSelectionResult(
                success=result['success'],
                selected_features=result.get('selected_features', pd.DataFrame()),
                selection_metadata=result.get('selection_metadata', {}),
                selection_metrics=result.get('selection_metrics', {}),
                artifacts=result.get('artifacts', {}),
                error_message=result.get('error_message')
            )
            
            if selection_result.success:
                self.logger.info(f"✅ Feature selection completed successfully with {len(selection_result.selected_features.columns)} selected features")
            else:
                self.logger.error(f"❌ Feature selection failed: {selection_result.error_message}")
            
            return selection_result
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection step failed with exception: {e}")
            return FeatureSelectionResult(
                success=False,
                selected_features=pd.DataFrame(),
                selection_metadata={},
                selection_metrics={},
                artifacts={},
                error_message=str(e)
            )


# Command handler for ares_launcher integration
async def handle_feature_generation_feature_selection_step(
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
) -> FeatureSelectionResult:
    """
    Handle feature generation feature selection step command.
    
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
        FeatureSelectionResult with selection results
    """
    # Create sample data for feature selection (in real usage, this would come from data loading)
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Generate targets using the labeling system
    from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import ConsolidatedPipelineRunner
    runner = ConsolidatedPipelineRunner()
    targets = runner._generate_targets(sample_data, symbol, timeframe, direction)
    
    # Create step instance and execute
    step = FeatureGenerationFeatureSelectionStep()
    
    return await step.execute(
        data=sample_data,
        targets=targets,
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