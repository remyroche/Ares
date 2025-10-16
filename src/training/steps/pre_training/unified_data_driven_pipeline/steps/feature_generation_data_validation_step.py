"""
Feature Generation Data Validation Step

This step performs comprehensive data validation and quality assessment
as the first step in the unified data-driven pipeline by calling the
consolidated pipeline at the appropriate stage.
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
    run_data_validation_step
)
from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent, ComponentConfig, ComponentResult
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

# Import tprint utilities for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_performance, tprint_step, tprint_result
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_step(*args, **kwargs): print("STEP:", *args, **kwargs)
    def tprint_result(*args, **kwargs): print("RESULT:", *args, **kwargs)

@dataclass
class DataValidationResult:
    """Result of data validation step."""

    success: bool
    data_quality_score: float
    validation_metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationDataValidationStep(BasePreTrainingComponent):
    """Data validation step that calls the consolidated pipeline."""

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the data validation step."""
        super().__init__(config or ComponentConfig())
        self.logger = logging.getLogger(__name__)

    async def execute(self,
                     training_input: Dict[str, Any],
                     pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Execute data validation step using consolidated pipeline."""

        self.logger.info("🔍 Starting data validation step using consolidated pipeline")

        # Extract parameters from training_input
        data = training_input.get('data')
        symbol = self.config.symbol
        timeframe = self.config.timeframe
        direction = training_input.get('direction', 'longs')
        intensity = training_input.get('intensity', 'blank')
        lookback_days = training_input.get('lookback_days')
        start_date = training_input.get('start_date')
        end_date = training_input.get('end_date')
        exchange = self.config.exchange
        custom_overrides = training_input.get('custom_overrides')

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

            # Convert result to ComponentResult
            component_result = ComponentResult(
                success=result['success'],
                artifacts=result.get('artifacts', {}),
                metadata={
                    'data_quality_score': result.get('data_quality_score', 0.0),
                    'validation_metadata': result.get('validation_metadata', {}),
                    **result.get('metadata', {})
                },
                error_message=result.get('error_message')
            )

            if component_result.success:
                self.logger.info(f"✅ Data validation completed successfully with quality score: {component_result.metadata.get('data_quality_score', 0):.3f}")
            else:
                self.logger.error(f"❌ Data validation failed: {component_result.error_message}")

            return component_result

        except Exception as e:
            self.logger.error(f"❌ Data validation step failed with exception: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={},
                error_message=str(e)
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

# Register component with factory
def _register_feature_generation_data_validation_step():
    """Register the feature generation data validation step component with the factory."""
    try:
        from ...components.component_factory import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_data_validation_step',
            FeatureGenerationDataValidationStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_data_validation_step()
