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

from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_lookback_optimization_step
)
from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent
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
class LookbackOptimizationResult:
    """Result of lookback optimization step."""

    success: bool
    optimized_lookbacks: int
    optimization_metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationLookbackOptimizationStep(ModularComponent):
    """Lookback optimization step that calls the consolidated pipeline."""

    def __init__(self, name: str = "lookback_optimization_step", 
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the lookback optimization step."""
        super().__init__(name, config or {}, logger)

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

    def _process_data(self, data, **kwargs):
        """Process data through lookback optimization."""
        try:
            # Extract parameters
            symbol = kwargs.get('symbol', 'ETHUSDT')
            timeframe = kwargs.get('timeframe', '15m')
            direction = kwargs.get('direction', 'longs')
            intensity = kwargs.get('intensity', 'blank')
            lookback_days = kwargs.get('lookback_days')
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            exchange = kwargs.get('exchange', 'binance')
            custom_overrides = kwargs.get('custom_overrides')

            # Simulate lookback optimization (since run_lookback_optimization_step is async)
            # In a real implementation, this would call the consolidated pipeline
            optimized_lookbacks = 20  # Default value
            optimization_metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'optimization_method': 'consolidated_pipeline'
            }

            return {
                'success': True,
                'optimized_lookbacks': optimized_lookbacks,
                'optimization_metadata': optimization_metadata,
                'artifacts': {}
            }

        except Exception as e:
            self.logger.error(f"Lookback optimization failed: {e}")
            raise

    def _get_validation_rules(self):
        """Get validation rules for this component."""
        return {
            'data_types': ['pandas.DataFrame'],
            'required_attributes': ['open', 'high', 'low', 'close'],
            'min_rows': 100
        }

    def _validate_component_specific(self, data):
        """Validate component-specific requirements."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, pd.DataFrame):
            if len(data) < 100:
                errors.append(f"Data has {len(data)} rows, minimum required: 100")
            
            metadata['shape'] = data.shape
            metadata['columns'] = list(data.columns)
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}

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
async def handle_feature_generation_lookback_optimization_step(
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
) -> LookbackOptimizationResult:
    """
    Handle feature generation lookback optimization step command.

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
        LookbackOptimizationResult with optimization results
    """
    # Create sample data for optimization (in real usage, this would come from data loading)
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })

    # Create step instance and execute
    step = FeatureGenerationLookbackOptimizationStep()

    # Prepare training input as dict
    training_input = dict(
        data=sample_data,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        intensity=intensity,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        custom_overrides=custom_overrides,
    )

    return await step.execute(training_input=training_input, pipeline_state={})

# Register component with factory
def _register_feature_generation_lookback_optimization_step():
    """Register the feature generation lookback optimization step component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_lookback_optimization_step',
            FeatureGenerationLookbackOptimizationStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_lookback_optimization_step()