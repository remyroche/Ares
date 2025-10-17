"""
Feature Generation Interaction Generation Step

This step generates feature interactions as part of the unified data-driven pipeline
by calling the consolidated pipeline at the appropriate stage.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_interaction_generation_step
)
from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe


@dataclass
class InteractionGenerationResult:
    """Result of interaction generation step."""

    success: bool
    generated_interactions: int
    interaction_metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationInteractionGenerationStep(ModularComponent):
    """Interaction generation step that calls the consolidated pipeline."""

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the interaction generation step."""
        # Convert ComponentConfig to dict for ModularComponent
        config_dict = config.to_dict() if config else {}
        super().__init__(
            name="feature_generation_interaction_generation_step",
            config=config_dict,
            logger=logging.getLogger(__name__)
        )

    async def execute(self,
                     training_input: Dict[str, Any],
                     pipeline_state: Dict[str, Any]) -> InteractionGenerationResult:
        """Execute interaction generation step using consolidated pipeline."""

        self.logger.info("🔧 Starting interaction generation step using consolidated pipeline")

        # Extract parameters from training_input
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
            result = await run_interaction_generation_step(
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

            # Validate result structure
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict result, got {type(result)}")
            
            # Safely extract values with defaults
            success = result.get('success', False)
            generated_interactions = result.get('generated_interactions', 0)
            interaction_metadata = result.get('interaction_metadata', {})
            artifacts = result.get('artifacts', {})
            error_message = result.get('error_message')

            # Convert result to InteractionGenerationResult
            interaction_result = InteractionGenerationResult(
                success=success,
                generated_interactions=generated_interactions,
                interaction_metadata=interaction_metadata,
                artifacts=artifacts,
                error_message=error_message
            )

            if interaction_result.success:
                self.logger.info(f"✅ Interaction generation completed successfully with {interaction_result.generated_interactions} generated interactions")
            else:
                self.logger.error(f"❌ Interaction generation failed: {interaction_result.error_message}")

            return interaction_result

        except Exception as e:
            self.logger.exception(f"❌ Interaction generation step failed with exception: {e}")
            return InteractionGenerationResult(
                success=False,
                generated_interactions=0,
                interaction_metadata={},
                artifacts={},
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

    # Required abstract methods from ModularComponent
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        try:
            # Set initial state
            self.set_state('initialized_at', datetime.now().isoformat())
            self.set_state('interaction_count', 0)
            return True
        except Exception as e:
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        self.set_state('cleaned_up_at', datetime.now().isoformat())
        self.set_state('interaction_count', 0)
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component logic."""
        # Increment interaction count
        count = self.get_state('interaction_count', 0)
        self.set_state('interaction_count', count + 1)
        
        # Basic processing - return data as-is for now
        # The actual interaction generation is done in the execute method
        return data
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        return {
            'min_size': 100,
            'max_size': 1000000,
            'required_attributes': ['open', 'high', 'low', 'close', 'volume'],
            'data_types': ['pandas.DataFrame'],
            'max_nan_ratio': 0.1,
            'min_unique_values': 2
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, pd.DataFrame):
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check data size
            if len(data) < 100:
                warnings.append("Data size is small (< 100 rows)")
            
            # Check for NaN values
            nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if nan_ratio > 0.1:
                warnings.append(f"High NaN ratio: {nan_ratio:.2%}")
            
            metadata['data_shape'] = data.shape
            metadata['nan_ratio'] = nan_ratio
            metadata['columns'] = list(data.columns)
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}

# Command handler for ares_launcher integration
async def handle_feature_generation_interaction_generation_step(
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
) -> InteractionGenerationResult:
    """
    Handle feature generation interaction generation step command.

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
        InteractionGenerationResult with generation results
    """
    # Only generate sample data if not provided
    data = kwargs.get('data')
    if data is None:
        # Create sample data for generation (in real usage, this would come from data loading)
        sample_data = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 105,
            'low': np.random.randn(1000).cumsum() + 95,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000)
        })
    else:
        # Validate provided data
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(data)}")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        sample_data = data

    # Validate mutually exclusive date parameters
    if lookback_days is not None and (start_date is not None or end_date is not None):
        raise ValueError("Cannot specify both lookback_days and start_date/end_date")

    # Create step instance and execute with proper signature
    step = FeatureGenerationInteractionGenerationStep()
    
    # Build training_input dict and pass explicit pipeline_state
    training_input = {
        'data': sample_data,
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
    
    pipeline_state = {}

    return await step.execute(training_input, pipeline_state)

# Register component with factory
def _register_feature_generation_interaction_generation_step():
    """Register the feature generation interaction generation step component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_interaction_generation_step',
            FeatureGenerationInteractionGenerationStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_interaction_generation_step()