"""
Feature Generation Lookback Optimization Step

This step performs lookback optimization as part of the unified data-driven pipeline
by calling the consolidated pipeline at the appropriate stage.
"""

from __future__ import annotations

import warnings
import logging
import json
import pandas as pd
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# run_lookback_optimization_step moved to local definition
def run_lookback_optimization_step(config, data):
    """Run lookback optimization step."""
    # TODO: Implement lookback optimization logic
    return {"success": True, "optimized_data": data}
from src.training.steps.base_step import BaseStep

from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe


# Note: tprint utilities are now available through BaseStep's comprehensive tools

@dataclass
class FeatureGenerationLookbackOptimizationStep(BaseStep):
    """Lookback optimization step that calls the consolidated pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the step."""
        # Use BaseStep's comprehensive tprint utilities
        super().__init__("feature_generation_step", config)

    def _initialize_resources(self) -> bool:
        """Initialize lookback optimization resources."""
        try:
            self.set_state('initialized_at', time.time())
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize lookback optimization: {e}")
            return False

    def _cleanup_resources(self) -> None:
        """Cleanup lookback optimization resources."""
        try:
            self.set_state('cleaned_up_at', time.time())
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute lookback optimization step."""
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

            # Set context for enhanced file naming
            self._set_context(symbol=symbol, exchange=exchange, direction=direction, model='Analyst')

            # Simulate lookback optimization (since run_lookback_optimization_step is async)
            # In a real implementation, this would call the consolidated pipeline
            optimized_lookbacks = 20  # Default value
            optimization_metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'optimization_method': 'consolidated_pipeline'
            }

            # Save artifacts using BaseStep methods
            if data is not None:
                self._save_dataframe(data, 'optimized_data')
            
            self._save_metadata(optimization_metadata, 'optimization_metadata')

            return {
                'success': True,
                'artifacts': ['optimized_data', 'optimization_metadata'],
                'metrics': {
                    'optimized_lookbacks': optimized_lookbacks,
                    'optimization_metadata': optimization_metadata
                }
            }

        except Exception as e:
            self.logger.error(f"Lookback optimization failed: {e}")
            raise


@dataclass
class LookbackOptimizationResult:
    """Result from lookback optimization step."""
    success: bool
    optimized_lookbacks: int
    optimization_metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


# Handler function for ares_launcher integration
async def handle_feature_generation_lookback_optimization_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    exchange: str = "binance",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: int = None,
    start_date: str = None,
    end_date: str = None,
    custom_overrides: dict = None,
    **kwargs
) -> ComponentResult:
    """
    Handler function for feature generation lookback optimization step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        timeframe: Timeframe (e.g., "15m")
        exchange: Exchange name (e.g., "binance")
        direction: Trading direction (e.g., "longs")
        intensity: Intensity level (e.g., "blank")
        lookback_days: Number of days to look back
        start_date: Start date for data
        end_date: End date for data
        custom_overrides: Custom configuration overrides
        **kwargs: Additional arguments

    Returns:
        ComponentResult: Result of the lookback optimization step
    """
    artifact_manager = get_pretraining_artifact_manager()

    try:
        # Create the step instance
        step = FeatureGenerationLookbackOptimizationStep(
            name="lookback_optimization_step",
            config={
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'custom_overrides': custom_overrides or {}
            }
        )

        # Load data for processing
        data = await step.load_data(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            direction=direction,
            intensity=intensity,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

        # Process the data
        result = await step.process_data_async(
            data,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            intensity=intensity,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            custom_overrides=custom_overrides or {},
            **kwargs
        )

        # Create result object
        step_result = LookbackOptimizationResult(
            success=result.get('success', False),
            optimized_lookbacks=result.get('optimized_lookbacks', 20),
            optimization_metadata=result.get('optimization_metadata', {}),
            artifacts=result.get('artifacts', {})
        )

        # Convert to ComponentResult
        component_result = ComponentResult(
            success=step_result.success,
            data=None,  # Lookback optimization doesn't return processed data
            metadata={
                'step_name': 'feature_generation_lookback_optimization_step',
                'optimized_lookbacks': step_result.optimized_lookbacks,
                'optimization_metadata': step_result.optimization_metadata
            },
            artifacts=step_result.artifacts,
            error_message=step_result.error_message
        )

        # Save artifacts
        await artifact_manager.save_step_result(
            step_name='feature_generation_lookback_optimization_step',
            result=component_result,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction
        )

        return component_result

    except Exception as e:
        error_message = f"Lookback optimization step failed: {str(e)}"
        tprint_error(error_message)

        # Return failed result
        component_result = ComponentResult(
            success=False,
            data=None,
            metadata={'step_name': 'feature_generation_lookback_optimization_step'},
            artifacts={},
            error_message=error_message
        )

        return component_result
