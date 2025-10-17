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
class FeatureGenerationInteractionGenerationStep(ModularComponent):
    """Interaction generation step that calls the consolidated pipeline."""

    def __init__(self, name: str = "step", config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """Initialize the interaction generation step."""
        super().__init__(name, config or {}, logger)
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
