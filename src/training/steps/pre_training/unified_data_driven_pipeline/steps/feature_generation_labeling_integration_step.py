"""
Feature Generation Labeling Integration Step

This step performs labeling integration as part of the unified data-driven pipeline
by calling the consolidated pipeline at the appropriate stage.
"""

from __future__ import annotations

import warnings
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
class LabelingIntegrationResult:
    """Result of labeling integration step."""

    success: bool
    integrated_labels: int
    integration_metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationLabelingIntegrationStep(BasePreTrainingComponent):
    """Labeling integration step that calls the consolidated pipeline."""

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the labeling integration step."""
        super().__init__(config or ComponentConfig())
        self.logger = logging.getLogger(__name__)

    async def execute(self,
                     training_input: Optional[Dict[str, Any]] = None,
                     pipeline_state: Optional[Dict[str, Any]] = None,
                     data: Optional[Any] = None,
                     **kwargs) -> ComponentResult:
        """Execute labeling integration step using consolidated pipeline."""

        self.logger.info("🔗 Starting labeling integration step using consolidated pipeline")

        # Extract parameters from training_input or kwargs
        if training_input is None:
            # Extract from kwargs (called from component factory)
            data = data or kwargs.get('data')
            symbol = kwargs.get('symbol', 'ETHUSDT')
            timeframe = kwargs.get('timeframe', '15m')
            direction = kwargs.get('direction', 'longs')
            intensity = kwargs.get('intensity', 'blank')
            lookback_days = kwargs.get('lookback_days')
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            exchange = kwargs.get('exchange', 'binance')
            custom_overrides = kwargs.get('custom_overrides')
            # Create training_input dict for compatibility
            training_input = {
                'data': data,
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
        else:
            # Extract from training_input (called from pipeline)
            data = training_input.get('data')
            symbol = training_input.get('symbol', 'ETHUSDT')
            timeframe = training_input.get('timeframe', '15m')
            direction = training_input.get('direction', 'longs')
            intensity = training_input.get('intensity', 'blank')
            lookback_days = training_input.get('lookback_days')
            start_date = training_input.get('start_date')
            end_date = training_input.get('end_date')
            exchange = training_input.get('exchange', 'binance')
            custom_overrides = training_input.get('custom_overrides')

        try:
            # Call the consolidated pipeline runner
            result = await run_labeling_integration_step(
                data,
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
                metadata={
                    'integrated_labels': result.get('integrated_labels', 0),
                    'integration_metadata': result.get('integration_metadata', {}),
                    'artifacts': result.get('artifacts', {}),
                    **result.get('metadata', {})
                },
                error_message=result.get('error_message')
            )

            if component_result.success:
                self.logger.info(f"✅ Labeling integration completed successfully with {component_result.metadata.get('integrated_labels', 0)} integrated labels")
            else:
                self.logger.error(f"❌ Labeling integration failed: {component_result.error_message}")

            return component_result

        except Exception as e:
            self.logger.error(f"❌ Labeling integration step failed with exception: {e}")
            return ComponentResult(
                success=False,
                metadata={'artifacts': {}},
                error_message=str(e)
            )

    # Required abstract methods from BasePreTrainingComponent
    def process(self, data: Any) -> Any:
        """Process the input data and return the result."""
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🔗 Processing labeling integration step")
            else:
                print("INFO: Processing labeling integration step")
            
            # Convert data to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ Input data is not a DataFrame, attempting conversion")
                else:
                    print("WARNING: Input data is not a DataFrame, attempting conversion")
                data = pd.DataFrame(data)
            
            # Basic validation
            if data.empty:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Input data is empty")
                else:
                    print("ERROR: Input data is empty")
                return None
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Processed {len(data)} rows for labeling integration")
            else:
                print(f"SUCCESS: Processed {len(data)} rows for labeling integration")
            
            return data
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Error processing labeling integration data: {e}")
            else:
                print(f"ERROR: Error processing labeling integration data: {e}")
            return None

    def validate(self, data: Any) -> bool:
        """Validate the input data for labeling integration."""
        try:
            if data is None:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Validation failed: Data is None")
                else:
                    print("ERROR: Validation failed: Data is None")
                return False
            
            if not isinstance(data, pd.DataFrame):
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Validation failed: Data is not a DataFrame")
                else:
                    print("ERROR: Validation failed: Data is not a DataFrame")
                return False
            
            if data.empty:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Validation failed: Data is empty")
                else:
                    print("ERROR: Validation failed: Data is empty")
                return False
            
            # Check for required columns for labeling
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ Validation failed: Missing required columns: {missing_columns}")
                else:
                    print(f"ERROR: Validation failed: Missing required columns: {missing_columns}")
                return False
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Validation passed: Data has {len(data)} rows with required columns")
            else:
                print(f"SUCCESS: Validation passed: Data has {len(data)} rows with required columns")
            
            return True
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Validation error: {e}")
            else:
                print(f"ERROR: Validation error: {e}")
            return False

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
        LabelingIntegrationResult with integration results
    """
    # Create sample data for integration (in real usage, this would come from data loading)
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

# Register component with factory
def _register_feature_generation_labeling_integration_step():
    """Register the feature generation labeling integration step component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_labeling_integration_step',
            FeatureGenerationLabelingIntegrationStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_labeling_integration_step()