"""
Feature Generation Period Optimization Step

This step performs period optimization as part of the unified data-driven pipeline
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
    run_period_optimization_step
)


@dataclass
class PeriodOptimizationResult:
    """Result of period optimization step."""
    
    success: bool
    optimal_periods: Dict[str, int]
    optimization_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


class FeatureGenerationPeriodOptimizationStep:
    """Period optimization step that calls the consolidated pipeline."""
    
    def __init__(self):
        """Initialize the period optimization step."""
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
                     custom_overrides: Optional[Dict[str, Any]] = None) -> PeriodOptimizationResult:
        """Execute period optimization step using consolidated pipeline."""
        
        self.logger.info("⏱️ Starting period optimization step using consolidated pipeline")
        
        try:
            # Call the consolidated pipeline runner
            result = await run_period_optimization_step(
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
            
            # Convert result to PeriodOptimizationResult
            optimization_result = PeriodOptimizationResult(
                success=result['success'],
                optimal_periods=result.get('optimal_periods', {}),
                optimization_metrics=result.get('optimization_metrics', {}),
                artifacts=result.get('artifacts', {}),
                error_message=result.get('error_message')
            )
            
            if optimization_result.success:
                self.logger.info(f"✅ Period optimization completed successfully with {len(optimization_result.optimal_periods)} optimized periods")
            else:
                self.logger.error(f"❌ Period optimization failed: {optimization_result.error_message}")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ Period optimization step failed with exception: {e}")
            return PeriodOptimizationResult(
                success=False,
                optimal_periods={},
                optimization_metrics={},
                artifacts={},
                error_message=str(e)
            )


# Command handler for ares_launcher integration
async def handle_feature_generation_period_optimization_step(
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
) -> PeriodOptimizationResult:
    """
    Handle feature generation period optimization step command.
    
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
        PeriodOptimizationResult with optimization results
    """
    # Create sample data for period optimization (in real usage, this would come from data loading)
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
    step = FeatureGenerationPeriodOptimizationStep()
    
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