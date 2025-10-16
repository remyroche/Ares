"""
Feature Generation Feature Generation Step

This step generates features as part of the unified data-driven pipeline
by calling the consolidated pipeline at the appropriate stage.
"""

from __future__ import annotations

import logging
import json
import warnings
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_feature_generation_step
)

from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent, ComponentConfig, ComponentResult
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

@dataclass
class FeatureGenerationResult:
    """Result of feature generation step."""

    success: bool
    generated_features: pd.DataFrame
    feature_metadata: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationFeatureGenerationStep(BasePreTrainingComponent):
    """Feature generation step that calls the consolidated pipeline."""

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the feature generation step."""
        super().__init__(config or ComponentConfig())
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
                     custom_overrides: Optional[Dict[str, Any]] = None) -> FeatureGenerationResult:
        """Execute feature generation step using consolidated pipeline."""

        self.logger.info("🔧 Starting feature generation step using consolidated pipeline")

        try:
            # Call the consolidated pipeline runner
            result = await run_feature_generation_step(
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

            # Convert result to FeatureGenerationResult
            generation_result = FeatureGenerationResult(
                success=result['success'],
                generated_features=result.get('generated_features', pd.DataFrame()),
                feature_metadata=result.get('feature_metadata', {}),
                generation_metrics=result.get('generation_metrics', {}),
                artifacts=result.get('artifacts', {}),
                error_message=result.get('error_message')
            )

            if generation_result.success:
                self.logger.info(f"✅ Feature generation completed successfully with {len(generation_result.generated_features.columns)} features")
            else:
                self.logger.error(f"❌ Feature generation failed: {generation_result.error_message}")

            return generation_result

        except Exception as e:
            self.logger.error(f"❌ Feature generation step failed with exception: {e}")
            return FeatureGenerationResult(
                success=False,
                generated_features=pd.DataFrame(),
                feature_metadata={},
                generation_metrics={},
                artifacts={},
                error_message=str(e)
            )

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
) -> FeatureGenerationResult:
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
        FeatureGenerationResult with generation results
    """
    # Create sample data for feature generation (in real usage, this would come from data loading)
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })

    # Create step instance and execute
    step = FeatureGenerationFeatureGenerationStep()

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

    # Required utility methods for BasePreTrainingComponent
    def safe_dataframe_operation(self, operation_func, *args, **kwargs):
        """Safe dataframe operation wrapper."""
        return safe_dataframe_operation(operation_func, *args, **kwargs)

    def safe_matrix_multiply(self, a, b):
        """Safe matrix multiplication."""
        return safe_matrix_multiply(a, b)

    def optimize_dataframe_for_matrix_ops(self, df):
        """Optimize dataframe for matrix operations."""
        return optimize_dataframe(df)

# Register component with factory
def _register_feature_generation_feature_generation_step():
    """Register the FeatureGenerationFeatureGenerationStep component with the factory."""
    try:
        from ...components.component_factory import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_feature_generation_step',
            FeatureGenerationFeatureGenerationStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_feature_generation_step()
